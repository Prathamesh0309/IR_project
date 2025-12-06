import matplotlib.pyplot as plt
import pandas as pd

class ExploratoryAnalysis:
    def __init__(self, df):
        self.df = df.copy()

    def convert_dates(self, date_column="Created Date"):
        '''
        Convert the specified date column to datetime and extract year, month, day of week.
        If date_column is None or not found, attempts to auto-detect a suitable date column'''
        # Auto-detect a sensible date column if not provided or missing
        if date_column is None or date_column not in self.df.columns:
            candidates = ['Created Date', 'created_date', 'CREATED DATE', 'Updated Date', 'UPDATED_DATE', 'Updated_Date', 'updated_date', 'BATCH_DATE', 'Updated Date']
            found = None
            for c in candidates:
                if c in self.df.columns:
                    found = c
                    break
            if found is None:
                # try fuzzy match by lowering
                for c in self.df.columns:
                    if 'created' in c.lower() or 'update' in c.lower() or 'date' in c.lower() and 'time' not in c.lower():
                        found = c
                        break
            if found is None:
                raise KeyError(f"No suitable date column found. Available columns: {list(self.df.columns)}")
            date_column = found

        self.df[date_column] = pd.to_datetime(self.df[date_column], errors='coerce')
        self.df['Year'] = self.df[date_column].dt.year
        self.df['Month'] = self.df[date_column].dt.month
        self.df['DayOfWeek'] = self.df[date_column].dt.day_name()
        # store the date column used for plotting
        self.date_column = date_column
        return self.df

    def plot_complaints_over_time(self, date_column="Created Date"):
        '''
        Plot the number of complaints over time (monthly).'''
        # Use provided date_column or the one selected during convert_dates
        date_col = date_column if date_column in self.df.columns else getattr(self, 'date_column', date_column)
        df_time = self.df.groupby(pd.Grouper(key=date_col, freq="M")).size()
        fig, ax = plt.subplots(figsize=(12,5))
        df_time.plot(ax=ax)
        ax.set_title("Monthly Complaints Trend")
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Complaints")
        return fig

    def complaints_by_dayofweek(self):
        '''
        Plot the number of complaints by day of the week.'''

        # Ensure DayOfWeek column exists
        df_days = self.df.groupby('DayOfWeek').size().reindex(
            ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        )
        fig, ax = plt.subplots(figsize=(10,5))
        df_days.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title("Complaints by Day of Week")
        ax.set_ylabel("Count")
        return fig

    def detect_group_column(self):
        """Try to detect a sensible grouping column for citizen groups or service categories.
        Returns column name or None.
        """
        candidates = ['Agency Name', 'Complaint Type', 'full_text', 'cluster']
        for c in candidates:
            if c in self.df.columns:
                return c
        # fallback to first categorical-like column
        for c in self.df.columns:
            if self.df[c].dtype == object and self.df[c].nunique() < 200:
                return c
        return None

    def top_groups(self, group_column=None, top_n=10):
        '''
        Return top N groups by count from the specified or detected group column.    
        '''
        gc = group_column if group_column is not None else self.detect_group_column()
        if gc is None:
            raise KeyError('No suitable group column found')
        counts = self.df[gc].value_counts().head(top_n)
        return gc, counts

    def seasonal_analysis(self, categories, date_column=None):
        """Return a DataFrame of counts of categories per season.
        categories: list of category values to include (e.g., top complaint types)
        """
        date_col = date_column if date_column in self.df.columns else getattr(self, 'date_column', None)
        if date_col is None:
            raise KeyError('No date column available for seasonal analysis')
        df = self.df.copy()
        df['Month'] = pd.to_datetime(df[date_col], errors='coerce').dt.month
        # Map month to season
        def month_to_season(m):
            if pd.isna(m):
                return None
            m = int(m)
            if m in [12,1,2]:
                return 'Winter'
            if m in [3,4,5]:
                return 'Spring'
            if m in [6,7,8]:
                return 'Summer'
            return 'Fall'
        df['Season'] = df['Month'].apply(month_to_season)
        return df

    def plot_dashboard(self, date_column=None, group_column=None, top_n=6):
        """Create a 2x2 dashboard figure: monthly trend, dayofweek, top groups, seasonal stack for top groups.
        Returns matplotlib Figure.
        """
        # Prepare date column
        date_col = date_column if date_column in self.df.columns else getattr(self, 'date_column', None)
        if date_col is None:
            raise KeyError('No date column found for dashboard')

        # Monthly trend
        df_time = self.df.groupby(pd.Grouper(key=date_col, freq='M')).size()

        # Day of week
        if 'DayOfWeek' not in self.df.columns:
            self.df['DayOfWeek'] = pd.to_datetime(self.df[date_col], errors='coerce').dt.day_name()
        df_days = self.df.groupby('DayOfWeek').size().reindex(
            ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        )

        # Top groups
        gc = group_column if group_column is not None else self.detect_group_column()
        if gc is None:
            top_groups = pd.Series([], dtype=int)
        else:
            top_groups = self.df[gc].value_counts().head(top_n)

        # Seasonal counts for top groups
        df_season = pd.to_datetime(self.df[date_col], errors='coerce')
        seasons = df_season.dt.month.map(lambda m: ('Winter' if m in [12,1,2] else 'Spring' if m in [3,4,5] else 'Summer' if m in [6,7,8] else 'Fall'))
        if gc is not None and not top_groups.empty:
            df_tmp = self.df[[gc]].copy()
            df_tmp['Season'] = seasons
            df_tmp = df_tmp[df_tmp[gc].isin(top_groups.index)]
            season_pivot = pd.crosstab(df_tmp['Season'], df_tmp[gc]).reindex(['Winter','Spring','Summer','Fall']).fillna(0)
        else:
            season_pivot = pd.DataFrame()

        # Build figure
        fig, axes = plt.subplots(2,2, figsize=(14,10))
        ax0 = axes[0,0]
        df_time.plot(ax=ax0, color='tab:blue')
        ax0.set_title('Monthly Complaints Trend')

        ax1 = axes[0,1]
        df_days.plot(kind='bar', ax=ax1, color='tab:orange')
        ax1.set_title('Complaints by Day of Week')

        ax2 = axes[1,0]
        if not top_groups.empty:
            top_groups.plot(kind='bar', ax=ax2, color='tab:green')
            ax2.set_title(f'Top {len(top_groups)} Groups ({gc})')
        else:
            ax2.text(0.5,0.5,'No group column found', ha='center')
            ax2.set_title('Top Groups')

        ax3 = axes[1,1]
        if not season_pivot.empty:
            season_pivot.plot(kind='bar', stacked=True, ax=ax3)
            ax3.set_title('Seasonal distribution of top groups')
        else:
            ax3.text(0.5,0.5,'No seasonal data', ha='center')
            ax3.set_title('Seasonal Analysis')

        plt.tight_layout()
        return fig

    def top_reasons_for_group(self, group_value, group_column=None, reason_column_candidates=None, top_n=5):
        """Return top reason strings (value_counts) for a specified group value.
        Tries a set of candidate columns for reason text.
        """
        gc = group_column if group_column is not None else self.detect_group_column()
        if gc is None:
            raise KeyError('No group column available')
        if reason_column_candidates is None:
            reason_column_candidates = ['BRIEF_DESCRIPTION', 'DETAILED_DESCRIPTION', 'brief_description', 'detailed_description', 'cleaned_text']
        df_sub = self.df[self.df[gc] == group_value]
        for rc in reason_column_candidates:
            if rc in df_sub.columns:
                return df_sub[rc].value_counts().head(top_n)
        return pd.Series([], dtype=object)

    def monthly_trend(self, date_column="Created Date"):
        """
        Cleanly computes monthly complaint counts from the given dataframe.
        Always robust to partial date corruption.
        """

        if date_column not in self.df.columns:
            raise KeyError(f"Date column '{date_column}' not found in dataframe.")

        # Step 1 — Force string first to avoid mixed types
        temp = self.df[date_column].astype(str)

        # Step 2 — Safely parse dates
        parsed = pd.to_datetime(temp, errors="coerce")

        # Step 3 — Drop invalid dates
        df_valid = self.df.copy()
        df_valid["parsed_date"] = parsed
        df_valid = df_valid.dropna(subset=["parsed_date"])

        # Step 4 — Extract year-month as period
        df_valid["year_month"] = df_valid["parsed_date"].dt.to_period("M")

        # Step 5 — Count
        monthly_counts = (
            df_valid["year_month"]
            .value_counts()
            .sort_index()
        )

        return monthly_counts

    def plot_monthly_trend(self, date_column="Created Date"):
        '''
        Plot monthly complaint trend from the given dataframe.
        '''
        monthly_counts = self.monthly_trend(date_column)

        plt.figure(figsize=(12,5))
        monthly_counts.plot(kind="line", marker="o")
        plt.title("Monthly Complaint Volume")
        plt.xlabel("Month")
        plt.ylabel("Number of Complaints")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

