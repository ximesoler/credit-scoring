import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from pathlib import Path


class DataProfile:
    """Generate data profiling and summaries"""
    
    def quick_profile(self, df, exclude=None):
        """Generate summary statistics for all features"""
        exclude = set(exclude or [])
        cols = [c for c in df.columns if c not in exclude]
        
        rows = []
        for col in cols:
            s = df[col]
            row = {
                "feature": col,
                "dtype": str(s.dtype),
                "missing_pct": round(100 * s.isna().mean(), 2),
                "n_unique": int(s.nunique(dropna=True)),
            }
            
            if pd.api.types.is_numeric_dtype(s):
                row.update({
                    "min": s.min(),
                    "median": s.median(),
                    "max": s.max(),
                })
            else:
                top = s.value_counts(dropna=True).head(1)
                if not top.empty:
                    row["top_category"] = str(top.index[0])
                    row["top_pct"] = round(100 * top.iloc[0] / s.notna().sum(), 2)
            
            rows.append(row)
        
        profile = pd.DataFrame(rows).sort_values(["missing_pct", "feature"],
                                                 ascending=[False, True]).reset_index(drop=True)
        return profile


class FeatureEngineering:
    """Handle feature engineering and preprocessing"""
    
    def make_pre_offer_features(self, df, keep_term=False, keep_target=True, drop_cols=None):
        """Filter to pre-origination features only"""
        drop_cols = drop_cols or set()
        
        dt_cols = set(df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns)
        name_date_cols = {c for c in df.columns if c.endswith("_d") or c.endswith("_date")}
        
        extra_dates = {"issue_d", "earliest_cr_line", "payment_plan_start_date", "year", "issue_year"}
        name_date_cols |= (extra_dates & set(df.columns))
        
        drop_cols |= dt_cols | name_date_cols
        
        cols_keep = [c for c in df.columns if c not in drop_cols]
        
        if not keep_term and "term" in cols_keep:
            cols_keep.remove("term")
        
        if not keep_target and "target" in cols_keep:
            cols_keep.remove("target")
        
        df_out = df[cols_keep].copy()
        df_out.attrs["dropped_columns"] = sorted([c for c in drop_cols if c in df.columns])
        
        return df_out
    
    def engineer_features(self, df):
        """Create derived features and handle data type conversions"""
        out = df.copy()
        
        if "emp_length" in out.columns:
            emp_map = {
                "10+ years": 10.0, "9 years": 9.0, "8 years": 8.0, "7 years": 7.0,
                "6 years": 6.0, "5 years": 5.0, "4 years": 4.0, "3 years": 3.0,
                "2 years": 2.0, "1 year": 1.0, "< 1 year": 0.5, "n/a": np.nan
            }
            out["emp_length_yrs"] = out["emp_length"].map(emp_map).astype("float32")
        
        if "revol_util" in out.columns:
            out["revol_util_pct"] = (
                out["revol_util"].astype(str)
                .str.replace("%", "", regex=False)
                .replace({"nan": np.nan, "None": np.nan, "": np.nan})
                .astype(float)
                .astype("float32")
            )
        
        if "fico_range_low" in out.columns and "fico_range_high" in out.columns:
            out["fico_mid"] = ((out["fico_range_low"] + out["fico_range_high"]) / 2).astype("float32")
        
        return out
    
    def handle_missing_values(self, df, prefix="", count_cols=None, no_rec_cols=None):
        """Impute missing values with domain-appropriate strategies"""
        out = df.copy()
        count_cols = count_cols or []
        no_rec_cols = no_rec_cols or []
        
        prefix_cols = [c for c in out.columns if c.startswith(prefix)]
        for col in prefix_cols + no_rec_cols:
            if col in out.columns:
                out[col] = out[col].fillna(999)
        
        for col in count_cols:
            if col in out.columns:
                out[col] = out[col].fillna(0)
        
        return out
    
    def winsorize_outliers(self, df, quantile_low=0.01, quantile_high=0.99):
        """Cap extreme values to reduce outlier impact"""
        out = df.copy()
        
        heavy_tail_features = [
            "annual_inc", "revol_bal", "avg_cur_bal", "tot_cur_bal",
            "tot_hi_cred_lim", "total_rev_hi_lim", "total_bal_ex_mort",
            "loan_amnt", "max_bal_bc"
        ]
        
        cols_to_clip = [c for c in heavy_tail_features if c in out.columns]
        
        if cols_to_clip:
            q = out[cols_to_clip].quantile([quantile_low, quantile_high])
            for col in cols_to_clip:
                out[col] = out[col].clip(q.loc[quantile_low, col], q.loc[quantile_high, col])
        
        return out


class StabilityMetrics:
    """Calculate PSI and stability metrics"""
    
    def calculate_psi_numeric(self, train_vals, valid_vals, bins=10):
        """Calculate PSI for numeric features"""
        train_series = pd.Series(train_vals).dropna()
        valid_series = pd.Series(valid_vals).dropna()
        
        if train_series.empty or valid_series.empty:
            return 0.0
        
        cuts = train_series.quantile(np.linspace(0, 1, bins + 1)).unique()
        if len(cuts) < 3:
            return 0.0
        
        train_binned = pd.cut(train_series, cuts, include_lowest=True)
        valid_binned = pd.cut(valid_series, cuts, include_lowest=True)
        
        p_train = train_binned.value_counts(normalize=True).sort_index().replace(0, 1e-6)
        p_valid = valid_binned.value_counts(normalize=True).reindex(p_train.index, fill_value=1e-6)
        
        return float(((p_train - p_valid) * np.log(p_train / p_valid)).sum())
    
    def calculate_psi_categorical(self, train_vals, valid_vals, top_k=20):
        """Calculate PSI for categorical features"""
        train_series = pd.Series(train_vals).astype(str)
        valid_series = pd.Series(valid_vals).astype(str)
        
        top_categories = train_series.value_counts().head(top_k).index
        train_series = train_series.where(train_series.isin(top_categories), "_OTHER_")
        valid_series = valid_series.where(valid_series.isin(top_categories), "_OTHER_")
        
        all_cats = train_series.value_counts(normalize=True).index.union(
            valid_series.value_counts(normalize=True).index
        )
        
        p_train = train_series.value_counts(normalize=True).reindex(all_cats, fill_value=0).replace(0, 1e-6)
        p_valid = valid_series.value_counts(normalize=True).reindex(all_cats, fill_value=0).replace(0, 1e-6)
        
        return float(((p_train - p_valid) * np.log(p_train / p_valid)).sum())
    
    def calculate_psi_scores(self, expected, actual, bins=10):
        """PSI for predicted scores"""
        cuts = np.quantile(expected, np.linspace(0, 1, bins + 1))
        cuts[0], cuts[-1] = -np.inf, np.inf
        
        exp_dist = np.histogram(expected, bins=cuts)[0] / len(expected)
        act_dist = np.histogram(actual, bins=cuts)[0] / len(actual)
        
        exp_dist = np.clip(exp_dist, 1e-6, None)
        act_dist = np.clip(act_dist, 1e-6, None)
        
        return np.sum((act_dist - exp_dist) * np.log(act_dist / exp_dist))


class WOEAnalysis:
    """Weight of Evidence and Information Value calculations"""
    
    def _prepare_bins(self, values, bins):
        """Normalize user bins for pd.cut"""
        v = pd.to_numeric(values, errors="coerce")
        vmin = np.nanmin(v)
        vmax = np.nanmax(v)
        eps = 1e-9
        
        arr = np.array(bins, dtype=float)
        arr = np.sort(np.unique(arr))
        
        if not np.isneginf(arr[0]):
            arr = np.insert(arr, 0, vmin - eps)
        if not np.isposinf(arr[-1]):
            arr = np.append(arr, vmax + eps)
        
        for i in range(1, len(arr)):
            if arr[i] <= arr[i-1]:
                arr[i] = np.nextafter(arr[i-1], np.inf)
        
        return arr
    
    def _bin_feature(self, s, bins=None, quantiles=10, top_k=50, is_categorical=False, missing_label="(missing)"):
        """Categorical: top-k + _OTHER_. Numeric: custom bins if given; else qcut"""
        s_in = s.copy()
        
        if is_categorical:
            ser = s_in.astype("string").fillna(missing_label)
            top = ser.value_counts().head(top_k).index
            return ser.where(ser.isin(top), "_OTHER_")
        
        s_in = pd.to_numeric(s_in, errors="coerce")
        
        if bins is not None:
            cuts = self._prepare_bins(s_in, bins)
            return pd.cut(s_in, bins=cuts, include_lowest=True, duplicates="drop")
        
        valid = s_in.dropna()
        if valid.empty:
            return pd.Series(np.nan, index=s_in.index)
        nunq = valid.nunique()
        if nunq < 2:
            v = valid.iloc[0]
            cuts = [v - 1e-9, v + 1e-9]
            return pd.cut(s_in, bins=cuts, include_lowest=True)
        
        q = min(quantiles, nunq)
        return pd.qcut(valid, q=q, duplicates="drop").reindex(s_in.index)
    
    def woe_table(self, df, feature, target, bins=None, quantiles=10, top_k=50, eps=1e-6, categorical=None):
        """WOE/IV table"""
        y = df[target].astype(int)
        if categorical is None:
            categorical = not pd.api.types.is_numeric_dtype(df[feature])
        
        groups = self._bin_feature(df[feature], bins=bins, quantiles=quantiles,
                                   top_k=top_k, is_categorical=categorical)
        
        tmp = pd.DataFrame({"grp": groups, "y": y}).dropna(subset=["grp"])
        if tmp.empty:
            empty = pd.DataFrame(columns=["Agrupacion","Total","G","B","%_bad_rate","WOE","IV_bin"])
            return empty, 0.0, empty.style
        
        agg = tmp.groupby("grp", dropna=False)["y"].agg(["count","sum"]).rename(
            columns={"count":"Total","sum":"B"}
        )
        agg["G"] = agg["Total"] - agg["B"]
        
        pct_g = (agg["G"] / max(agg["G"].sum(), eps)).replace(0, eps)
        pct_b = (agg["B"] / max(agg["B"].sum(), eps)).replace(0, eps)
        
        agg["%_bad_rate"] = agg["B"] / agg["Total"]
        agg["WOE"] = np.log(pct_g / pct_b)
        agg["IV_bin"] = (pct_g - pct_b) * agg["WOE"]
        iv_total = float(agg["IV_bin"].sum())
        
        agg = agg.reset_index().rename(columns={"grp":"Agrupacion"})
        if not categorical and pd.api.types.is_interval_dtype(agg["Agrupacion"]):
            agg = agg.sort_values(by=agg["Agrupacion"].apply(lambda x: x.left))
        
        cols = ["Agrupacion","Total","G","B","%_bad_rate","WOE","IV_bin"]
        agg = agg[cols]
        styled = (agg.style
                    .background_gradient(subset=["%_bad_rate"], cmap="RdYlGn_r")
                    .format({"%_bad_rate":"{:.1%}","WOE":"{:.6f}","IV_bin":"{:.6f}"}))
        return agg, iv_total, styled
    
    def show_woe_for_columns(self, df, target, num_cols=None, cat_cols=None,
                             num_bins=None, num_quantiles=10,
                             cat_topk=50, show_tables=False):
        """Calculate IV for multiple columns"""
        iv_summary = []
        num_cols = (num_cols or []).copy()
        cat_cols = (cat_cols or []).copy()
        num_bins = num_bins or {}
        
        cols = list(dict.fromkeys(num_cols + cat_cols))
        
        for col in cols:
            bins = num_bins.get(col)
            if bins is not None:
                tbl, iv_tot, st = self.woe_table(df, col, target, bins=bins, categorical=False)
            else:
                is_num = pd.api.types.is_numeric_dtype(df[col])
                tbl, iv_tot, st = self.woe_table(
                    df, col, target,
                    quantiles=num_quantiles,
                    top_k=cat_topk,
                    categorical=not is_num
                )
            
            if show_tables:
                print(col)
                try:
                    from IPython.display import display
                    display(st)
                except Exception:
                    pass
            
            iv_summary.append({
                "feature": col,
                "iv_total": iv_tot,
                "type": "numeric" if (bins is not None or pd.api.types.is_numeric_dtype(df[col])) else "categorical"
            })
        
        iv_df = pd.DataFrame(iv_summary).sort_values("iv_total", ascending=False)
        return iv_df


class ModelEvaluation:
    """Model evaluation metrics and analysis"""

    def ks_statistic(self, y_true, y_pred):
        """Kolmogorov-Smirnov statistic"""
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        return np.max(np.abs(tpr - fpr))

    def comprehensive_metrics(self, y_true, y_pred, dataset_name=""):
        """Calculate all evaluation metrics"""
        from sklearn.metrics import average_precision_score
        
        metrics = {
            f"AUC_{dataset_name}": roc_auc_score(y_true, y_pred),
            f"PR_AUC_{dataset_name}": average_precision_score(y_true, y_pred),
            f"KS_{dataset_name}": self.ks_statistic(y_true, y_pred),
        }
        return metrics
        
    def plot_ks_curve_deciles(self, y_true, y_pred, model_name, ax=None):
        """Plot K-S curve using decile approach (common in credit scoring)"""
        
        # Create dataframe
        df = pd.DataFrame({
            'score': y_pred,
            'target': y_true
        })
        
        # Sort by score descending and create deciles
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
        df['decile'] = pd.qcut(df['score'], q=10, labels=False, duplicates='drop') + 1
        
        # Calculate cumulative distributions by decile
        decile_stats = df.groupby('decile').agg({
            'target': ['sum', 'count']
        }).reset_index()
        decile_stats.columns = ['decile', 'defaults', 'total']
        decile_stats['non_defaults'] = decile_stats['total'] - decile_stats['defaults']
        
        # Cumulative percentages
        decile_stats['cum_defaults_pct'] = decile_stats['defaults'].cumsum() / decile_stats['defaults'].sum()
        decile_stats['cum_non_defaults_pct'] = decile_stats['non_defaults'].cumsum() / decile_stats['non_defaults'].sum()
        decile_stats['ks'] = np.abs(decile_stats['cum_defaults_pct'] - decile_stats['cum_non_defaults_pct'])
        
        max_ks = decile_stats['ks'].max()
        max_ks_decile = decile_stats.loc[decile_stats['ks'].idxmax(), 'decile']
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot
        ax.plot(decile_stats['decile'], decile_stats['cum_defaults_pct'] * 100, 
                marker='o', label='% Defaults Captured', linewidth=2, color='red')
        ax.plot(decile_stats['decile'], decile_stats['cum_non_defaults_pct'] * 100,
                marker='s', label='% Non-Defaults Captured', linewidth=2, color='blue')
        ax.plot(decile_stats['decile'], decile_stats['ks'] * 100,
                marker='^', label='K-S', linewidth=2, color='green', linestyle='--')
        
        ax.axhline(max_ks * 100, color='green', linestyle=':', alpha=0.5)
        ax.axvline(max_ks_decile, color='green', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Score Decile (1=Highest Risk, 10=Lowest Risk)', fontsize=12)
        ax.set_ylabel('Cumulative %', fontsize=12)
        ax.set_title(f'{model_name} - K-S Curve by Decile (Max K-S = {max_ks:.3f})', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 11))
        
        return max_ks, max_ks_decile
    
    def create_scorecard_table(self, y_true, y_pred, model_name="Model"):
        """
        Create traditional credit scorecard table with decile analysis
        """
        df = pd.DataFrame({
            'score': y_pred,
            'target': y_true
        })
        
        # Sort by score ASCENDING (lowest risk first) and create deciles
        df = df.sort_values('score', ascending=True).reset_index(drop=True)
        df['decile'] = pd.qcut(df['score'], q=10, labels=False, duplicates='drop') + 1
        
        decile_stats = []
        
        total_goods = (y_true == 0).sum()
        total_bads = (y_true == 1).sum()
        total_population = len(y_true)
        
        cum_goods = 0
        cum_bads = 0
        cum_population = 0
        
        for decile in range(1, 11):
            decile_data = df[df['decile'] == decile]
            
            n_total = len(decile_data)
            n_bads = (decile_data['target'] == 1).sum()
            n_goods = (decile_data['target'] == 0).sum()
            
            cum_population += n_total
            cum_goods += n_goods
            cum_bads += n_bads
            
            # Calculate metrics
            pct_population = (n_total / total_population) * 100
            pct_cum_population = (cum_population / total_population) * 100
            
            pct_goods = (cum_goods / total_goods) * 100
            pct_bads = (cum_bads / total_bads) * 100
            
            default_rate = (n_bads / n_total * 100) if n_total > 0 else 0
            
            # Odds = Goods / Bads
            odds = (n_goods / n_bads) if n_bads > 0 else 999
            
            # K-S = |% Goods - % Bads|
            ks = abs(pct_goods - pct_bads)
            
            # ICC (Information Capture Curve) - cumulative information value
            if n_bads > 0 and n_goods > 0:
                dist_bads = n_bads / total_bads
                dist_goods = n_goods / total_goods
                icc = (dist_bads - dist_goods) * np.log(dist_bads / dist_goods) if dist_goods > 0 else 0
            else:
                icc = 0
            
            # Score range for this decile
            min_score = decile_data['score'].min()
            max_score = decile_data['score'].max()
            
            decile_stats.append({
                'Variable': f'{min_score:.2%}-{max_score:.2%}',
                'Total_All': n_total,
                '%INT_All': pct_population,
                '%AC_All': pct_cum_population,
                'Total_Goods': n_goods,
                '%INT_Goods': (n_goods / total_goods * 100),
                '%AC_Goods': pct_goods,
                'Total_Bads': n_bads,
                '%INT_Bads': (n_bads / total_bads * 100),
                '%AC_Bads': pct_bads,
                'ODDS': odds,
                'K-S': ks,
                'ICC': icc,
            })
        
        result_df = pd.DataFrame(decile_stats)
        
        # Calculate summary metrics
        max_ks = result_df['K-S'].max()
        max_ks_decile = result_df['K-S'].idxmax() + 1
        gini = 2 * roc_auc_score(y_true, y_pred) - 1
        ar = roc_auc_score(y_true, y_pred)
        
        print(f"\n{'='*80}")
        print(f"{model_name} - Scorecard Analysis")
        print(f"{'='*80}")
        print(f"K-S:   {max_ks:.2f}%")
        print(f"AR:    {ar:.2%}")
        print(f"GINI:  {gini:.2%}")
        print(f"{'='*80}\n")
        
        return result_df


    def decile_ks_analysis(self, y_true, y_pred, model_name="Model"):

        result_df = self.create_scorecard_table(y_true, y_pred, model_name)
        """
        Display scorecard table with exact styling from the image
        """
        import pandas as pd
        
        # Create multi-level column structure
        display_data = []
        
        for idx, row in result_df.iterrows():
            display_data.append({
                ('Segment', 'Variable'): row['Variable'],
                ('All', 'Total'): int(row['Total_All']),
                ('All', '%INT'): f"{row['%INT_All']:.2f}%",
                ('All', '%AC'): f"{row['%AC_All']:.2f}%",
                ('Goods', 'Total'): int(row['Total_Goods']),
                ('Goods', '%INT'): f"{row['%INT_Goods']:.2f}%",
                ('Goods', '%AC'): f"{row['%AC_Goods']:.2f}%",
                ('Bads', 'Total'): int(row['Total_Bads']),
                ('Bads', '%INT'): row['%INT_Bads'],  # Keep as float for coloring
                ('Bads', '%AC'): f"{row['%AC_Bads']:.2f}%",
                ('Metrics', 'ODDS'): f"{row['ODDS']:.2f}",
                ('Metrics', 'K-S'): f"{row['K-S']:.2f}%",
            })
        
        display_df = pd.DataFrame(display_data)

        def color_gradient(val):
            """Apply gradient from green (low) to red (high) for default rates"""
            try:

                if isinstance(val, str):
                    val = float(val.replace('%', ''))
                
                if val < 1:
                    return 'background-color: #00551a; color: white'  # Dark green
                elif val < 2:
                    return 'background-color: #007a2d; color: white'  # Green
                elif val < 3:
                    return 'background-color: #00a83f; color: white'  # Medium green
                elif val < 5:
                    return 'background-color: #4db36f'  # Light green
                elif val < 7:
                    return 'background-color: #80c995'  # Very light green
                elif val < 10:
                    return 'background-color: #ffeb99'  # Light yellow
                elif val < 15:
                    return 'background-color: #ffcc66'  # Yellow-orange
                elif val < 20:
                    return 'background-color: #ff9933'  # Orange
                else:
                    return 'background-color: #cc0000; color: white'  # Dark red
            except:
                return ''
        
        styled = display_df.style.applymap(
            color_gradient, 
            subset=[('Bads', '%INT')]
        ).set_properties(**{
            'text-align': 'center',
            'border': '1px solid #ddd'
        }).set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', "#0a0909"),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('border', '1px solid #ddd')
            ]}
        ])
        
        def format_pct(val):
            try:
                if isinstance(val, (int, float)):
                    return f"{val:.2f}%"
                return val
            except:
                return val
        
        styled = styled.format(format_pct, subset=[('Bads', '%INT')])
        
        return styled


class BusinessAnalysis:
    """Business analysis and threshold optimization"""
    
    def analyze_thresholds(self, y_true, y_pred, avg_loan_amount=15000, default_loss_rate=0.80):
        """Analyze different approval thresholds"""
        thresholds = np.arange(0.05, 0.95, 0.05)
        results = []
        
        for thresh in thresholds:
            y_pred_class = (y_pred >= thresh).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
            
            total_applications = len(y_true)
            approved = (y_pred < thresh).sum()
            approval_rate = approved / total_applications
            
            approved_idx = y_pred < thresh
            if approved > 0:
                approved_defaults = y_true[approved_idx].sum()
                default_rate_approved = approved_defaults / approved
            else:
                approved_defaults = 0
                default_rate_approved = 0
            
            expected_loss = approved_defaults * avg_loan_amount * default_loss_rate
            
            results.append({
                "threshold": thresh,
                "approval_rate": approval_rate,
                "approved_count": approved,
                "default_rate_approved": default_rate_approved,
                "approved_defaults": approved_defaults,
                "expected_loss": expected_loss,
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0
            })
        
        return pd.DataFrame(results)
