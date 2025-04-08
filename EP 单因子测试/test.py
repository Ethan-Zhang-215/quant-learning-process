#导入基础数据库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.font_manager as fm
import seaborn as sns
import statsmodels.api as sm
from statsmodels.robust.robust_linear_model import RLM
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_rows', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

#============== 读取文件 ======================================================================================
#指数文件Index_df
Index_df = pd.read_csv('/Users/zhangyikun/Desktop/笔试/AIndexEODPrices.csv', parse_dates=['trade_dt'])
# 提取中证800收益率（假设涨跌幅字段为pct_chg）
zz800_returns = Index_df[Index_df['s_info_windcode'] == '000906.SH'][['trade_dt', 's_dq_pctchange']].copy()
zz800_returns['trade_dt'] = pd.to_datetime(zz800_returns['trade_dt'])

#读取合并后的文件
df= pd.read_csv('/Users/zhangyikun/Desktop/笔试/final_data.csv', parse_dates=['trade_dt'])
df['trade_dt'] = pd.to_datetime(df['trade_dt'])
#============== 数据处理 ======================================================================================
#  定义日期范围（确保转换为Timestamp）
start_date = pd.to_datetime('2024-01-01')
end_date = pd.to_datetime('2025-02-28')

# 筛选日期范围（包含边界）
date_mask = (df['trade_dt'] >= start_date) & (df['trade_dt'] <= end_date)
df = df.loc[date_mask].copy()  # 使用copy()避免SettingWithCopyWarning
zz800_mask = (zz800_returns['trade_dt'] >= start_date) & (zz800_returns['trade_dt'] <= end_date)
zz800_returns = zz800_returns.loc[zz800_mask].copy()

#统计股票中ST/PT股票数量，没有就不需要剔除
#st_mask = df['s_info_windcode'].str.contains('ST|PT', na=False)
#print(f"原始数据中的ST/PT股票数量: {st_mask.sum()}")

#剔除选股日停牌的股票，
first_days = df.groupby(pd.Grouper(key='trade_dt', freq='MS'))['trade_dt'].first().values
df['is_first_day'] = df.index.isin(first_days)
df = df[~((df['is_first_day']) & (df['s_dq_tradestatus'] == '停牌'))]

#MAD去极值（市盈率选取静态市盈率）
df['EP'] = 1 / df['s_val_pe_ttm']
median_ep = df['EP'].median()
mad_ep = np.median(np.abs(df['EP'] - median_ep))
upper_limit = median_ep + 3 * 1.4826 * mad_ep
lower_limit = median_ep - 3 * 1.4826 * mad_ep
df['EP_cleaned'] = df['EP'].clip(lower_limit, upper_limit)


#月度选股，每月计算缺失值比例，如果缺失率大于20%，作剔除处理，小于20%用行业中位数替代
# 计算每月缺失比例
df['month'] = df['trade_dt'].dt.to_period('M')
missing_ratio_by_month = df.groupby('month')['EP_cleaned'].apply(lambda x: x.isnull().mean())
#print("每月EP缺失值比例:\n", missing_ratio_by_month)
#每月缺失值比例都大于20%，全部剔除
df = df.dropna(subset=['EP_cleaned'])

#Z标准化
# 计算均值和标准差
mean_ep = df['EP_cleaned'].mean()
std_ep = df['EP_cleaned'].std()

# 标准化
df['EP_standardized'] = (df['EP_cleaned'] - mean_ep) / std_ep

# 测试两种市值（总市值和流通总市值）的因子IC，选择更显著者
ic_total = df['EP'].corr(np.log(df['s_val_mv']), method='spearman')
ic_float = df['EP'].corr(np.log(df['s_dq_mv']), method='spearman')

# 检查市值缺失比例，为0不作考虑
missing_ratio = df['s_dq_mv'].isnull().mean()

#============== 行业市值去中性化 ===============
#  对数市值
df['log_mcap'] = np.log(df['s_dq_mv'])

# 生成行业哑变量
industry_dummies = pd.get_dummies(df['industriescode'], prefix='ind')

# 回归中性化
X = pd.concat([df['log_mcap'], industry_dummies], axis=1)
y = df['EP_standardized']
model = LinearRegression()
model.fit(X, y)
df['EP_neutral'] = y - model.predict(X)

# ============== 因子收益显著性检验 (修正版) ====================================================================================
def factor_significance_test(data, factor_col='EP_standardized', ret_col='s_dq_pctchange'):
    """
    修正后的因子收益显著性检验
    主要修复了数据类型问题并增加了错误处理
    """
    # 准备月度横截面数据
    monthly_data = data.groupby(['month', 's_info_windcode']).last().reset_index()

    # 存储回归结果
    results = []

    # 按月进行横截面回归
    for month, month_data in monthly_data.groupby('month'):
        try:
            if len(month_data) < 10:  # 确保有足够样本量
                continue

            # 1. 准备回归数据（确保数值类型）
            X = month_data[[factor_col, 'log_mcap']].astype(float).copy()

            # 添加行业哑变量（确保是数值类型）
            industries = pd.get_dummies(month_data['industriescode'].astype(str), prefix='ind')
            X = pd.concat([X, industries.astype(float)], axis=1)

            # 添加截距项（必须是数值1.0）
            X['const'] = 1.0
            X = X.astype(float)  # 强制转换为float

            # 2. 准备因变量（确保是数值类型）
            y = month_data[ret_col].astype(float) / 100  # 收益率转换为小数

            # 3. 检查数据有效性
            if X.isnull().any().any() or y.isnull().any():
                print(f"Month {month}: 存在缺失值，跳过")
                continue

            # 4. 稳健回归（RLM）
            model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
            result = model.fit()

            # 5. 提取结果（因子列是第0个系数，const是最后一个）
            factor_return = result.params[0]
            t_value = result.tvalues[0]

            results.append({
                'month': month,
                'factor_return': factor_return,
                't_value': t_value,
                'n_stocks': len(month_data)
            })

        except Exception as e:
            print(f"Month {month} 处理失败: {str(e)}")
            continue

    if not results:
        raise ValueError("所有月份回归失败，请检查数据")

    # 转换为DataFrame
    results_df = pd.DataFrame(results).set_index('month')

    # 计算汇总统计量
    stats = pd.Series({
        'Mean Factor Return': results_df['factor_return'].mean(),
        'Std Factor Return': results_df['factor_return'].std(),
        'Mean Abs(t-value)': results_df['t_value'].abs().mean(),
        't-value > 2 Ratio': (results_df['t_value'].abs() > 2).mean(),
        'Positive Return Ratio': (results_df['factor_return'] > 0).mean(),
        'Positive t-value Ratio': (results_df['t_value'] > 0).mean(),
        'Avg Stocks per Month': results_df['n_stocks'].mean()
    })

    return results_df, stats


# ============== 执行检验 ==============
try:
    # 确保关键列是数值类型
    df['EP_standardized'] = pd.to_numeric(df['EP_standardized'], errors='coerce')
    df['s_dq_pctchange'] = pd.to_numeric(df['s_dq_pctchange'], errors='coerce')
    df['log_mcap'] = pd.to_numeric(df['log_mcap'], errors='coerce')

    # 删除缺失值
    df = df.dropna(subset=['EP_neutral', 's_dq_pctchange', 'log_mcap'])

    # 执行检验
    factor_results, factor_stats = factor_significance_test(df)

    # ============== 结果展示 ==============
    print("=" * 80)
    print("因子收益显著性检验汇总统计")
    print("=" * 80)
    print(factor_stats.to_frame('Value').round(4))

    plt.figure(figsize=(12, 6))
    factor_results['factor_return'].plot(kind='bar',
                                         color=np.where(factor_results['factor_return'] > 0, 'g', 'r'))
    plt.title('Monthly Factor Returns')
    plt.ylabel('Return')
    plt.savefig('factor_returns_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

except Exception as e:
    print(f"分析失败: {str(e)}")
    print("可能的原因:")
    print("1. 数据中存在非数值类型")
    print("2. 某些月份样本量不足")
    print("3. 行业变量包含异常值")




#============== IC指标计算 ==========================================================================
def calculate_ic(df):
    # 数据预处理：排序、去NaN
    df = df.sort_values(['s_info_windcode', 'trade_dt']).copy()
    ic_data = df[['EP_neutral', 's_info_windcode', 'month', 's_dq_pctchange']].dropna().copy()

    # 计算下月收益率（假设s_dq_pctchange已经是百分比，如5表示5%）
    ic_data['next_month_ret'] = ic_data.groupby('s_info_windcode')['s_dq_pctchange'].shift(-1)
    ic_data = ic_data.dropna(subset=['next_month_ret'])

    # 按月计算IC（要求每月至少10只股票）
    ic_results = ic_data.groupby('month').apply(
        lambda x: x['EP_neutral'].corr(x['next_month_ret'], method='spearman') if len(x) >= 10 else np.nan
    ).to_frame('IC').dropna()

    # 计算统计指标
    from scipy.stats import ttest_1samp
    t_stat, p_value = ttest_1samp(ic_results['IC'], 0)

    ic_stats = pd.Series({
        'IC Mean': ic_results['IC'].mean(),
        'IC Std': ic_results['IC'].std(),
        'IR': ic_results['IC'].mean() / ic_results['IC'].std(),
        'IC>0 Ratio': (ic_results['IC'] > 0).mean(),
        '|IC|>0.02 Ratio': (ic_results['IC'].abs() > 0.02).mean(),
        'IC p-value': p_value
    })

    return ic_results, ic_stats


# 调用函数
ic_results, ic_stats = calculate_ic(df)
print("IC统计指标:\n", ic_stats)

# 画图前处理Period类型索引
try:
    ic_results.index = pd.to_datetime(ic_results.index)
except TypeError:
    # 如果是Period类型，转换为Timestamp
    ic_results.index = ic_results.index.to_timestamp()

# 画图
plt.figure(figsize=(12, 4))
# 根据数据量选择图形类型
plot_kind = 'bar' if len(ic_results) < 12 else 'line'
ic_results['IC'].plot(kind=plot_kind)

# 添加参考线和标注
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axhline(y=ic_stats['IC Mean'], color='r', linestyle='--',
            label=f'Mean IC: {ic_stats["IC Mean"]:.3f} (IR={ic_stats["IR"]:.2f})')
plt.title(f'Monthly IC (p-value={ic_stats["IC p-value"]:.3f})')
plt.xlabel('Month')
plt.ylabel('IC Value')
plt.legend()
plt.grid(axis='y')
plt.tight_layout()  # 防止标签重叠
save_path = "IC_analysis.png"  # 可修改路径
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()





# ============== 权重设置方案 ================================================================================
# 1. 数据排序 - 确保按股票代码和交易日期排序
df = df.sort_values(['s_info_windcode', 'trade_dt']).copy()

# 2. 每月分组 - 基于EP_neutral值将股票分为10组
df['month'] = df['trade_dt'].dt.to_period('M')

# 获取每月每只股票的第一个EP_neutral值(选股日值)
monthly_groups = df.groupby(['month', 's_info_windcode'])['EP_neutral'].first().reset_index()

# 使用分位数分组(10组)，处理重复值情况
monthly_groups['group'] = monthly_groups.groupby('month')['EP_neutral'].transform(
    lambda x: pd.qcut(x, q=10, labels=range(1, 11), duplicates='drop')
)

# 3. 合并分组信息回原数据
df = df.merge(
    monthly_groups[['month', 's_info_windcode', 'group']],
    on=['month', 's_info_windcode'],
    how='left'
)

# 4. 计算每月每组的股票数量(用于等权配置)
group_counts = monthly_groups.groupby(['month', 'group']).size().reset_index(name='count')

# 5. 初始权重设置(每月第一个交易日)
# 标记每月每只股票的第一个交易日
df['is_month_start'] = df.groupby(['month', 's_info_windcode'])['trade_dt'].transform('min') == df['trade_dt']

# 合并组内股票数量信息
df = df.merge(group_counts, on=['month', 'group'], how='left')

# 设置初始权重:选股日等权重，非选股日设为NaN
df['weight'] = np.where(
    df['is_month_start'],
    1 / df['count'],  # 等权重
    np.nan
)


# 6. 改进的权重计算函数(考虑停牌和缺失值)
def calculate_weights(group):
    group = group.sort_values('trade_dt')
    current_weight = None

    for i, row in group.iterrows():
        # 如果是选股日，使用初始权重
        if pd.notna(row['weight']):
            current_weight = row['weight']
        # 如果已有权重且股票未停牌且有收益率数据
        elif (pd.notna(current_weight) and
              row['s_dq_tradestatus'] != '停牌' and
              pd.notna(row['s_dq_pctchange'])):
            # 按收益率调整权重(注意收益率是百分比，需要除以100)
            current_weight *= (1 + row['s_dq_pctchange'] / 100)

        # 更新权重(停牌股票保持原权重)
        group.at[i, 'weight'] = current_weight

    return group


# 对每只股票每月应用权重计算
df = df.groupby(['s_info_windcode', 'month'], group_keys=False).apply(calculate_weights)


# 7. 改进的权重归一化(考虑停牌股票)
def normalize_weights(group):
    # 只对非停牌且有权重的股票进行归一化
    valid_stocks = group[(group['s_dq_tradestatus'] != '停牌') & (group['weight'].notna())]
    total_weight = valid_stocks['weight'].sum()

    if total_weight > 0:
        # 归一化非停牌股票权重
        group.loc[valid_stocks.index, 'weight'] = valid_stocks['weight'] / total_weight
        # 停牌股票权重保持不变

    return group


# 每月每日每组进行权重归一化
df = df.groupby(['month', 'group', 'trade_dt'], group_keys=False).apply(normalize_weights)


#============== 分组指标计算 ==========================================================================
# 1. 计算组合收益率（修正百分比问题）
portfolio_ret = df.groupby([df['trade_dt'], 'group']).apply(
    lambda x: (x['weight'] * x['s_dq_pctchange'] / 100).sum()
).unstack()

# 2. 计算超额收益（确保基准收益率对齐）
benchmark_ret = zz800_returns.set_index('trade_dt')['s_dq_pctchange'] / 100
excess_ret = portfolio_ret.sub(benchmark_ret, axis=0)

# 3. 计算累计收益率（修正计算顺序）
cumulative_ret = (1 + portfolio_ret).cumprod() - 1
cumulative_excess_ret = (1 + excess_ret).cumprod() - 1


# 4. 改进的指标计算函数
def calculate_performance_metrics(returns, freq='D', is_excess=False):
    metrics = pd.DataFrame()

    # 确保输入为DataFrame
    if isinstance(returns, pd.Series):
        returns = pd.DataFrame(returns)

    # 年化因子
    ann_factor = 252 if freq == 'D' else 12

    # 年化收益率
    annual_ret = (1 + returns.mean()) ** ann_factor - 1

    # 年化波动率
    annual_vol = returns.std() * np.sqrt(ann_factor)

    # 夏普比率（无风险利率为0）
    sharpe = annual_ret / annual_vol

    # 最大回撤计算
    def calc_max_dd(series):
        cum = (1 + series).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak
        return dd.min()

    max_drawdown = returns.apply(calc_max_dd)

    # 信息比率（仅对超额收益计算）
    if is_excess:
        info_ratio = (returns.mean() / returns.std()) * np.sqrt(ann_factor)
    else:
        info_ratio = pd.Series(np.nan, index=returns.columns)

    metrics['Annualized Return'] = annual_ret
    metrics['Annualized Volatility'] = annual_vol
    metrics['Sharpe Ratio'] = sharpe
    metrics['Max Drawdown'] = max_drawdown
    metrics['Information Ratio'] = info_ratio

    return metrics


# 5. 计算指标（修正信息比率计算）
group_metrics = calculate_performance_metrics(portfolio_ret)
excess_metrics = calculate_performance_metrics(excess_ret, is_excess=True)

print("各组绝对收益指标:")
print(group_metrics)
print("\n各组超额收益指标:")
print(excess_metrics)

# ============== Visualization Section =================================================================

# 1. Plot group cumulative return curves
plt.figure(figsize=(12, 6))
for group in sorted(cumulative_ret.columns):
    plt.plot(cumulative_ret.index,
             cumulative_ret[group],
             label=f'Group {group}',
             linewidth=1.5)

# Add benchmark curve
plt.plot(cumulative_ret.index,
         (1 + benchmark_ret).cumprod() - 1,
         'k--',
         label='CSI 800',
         linewidth=2)

plt.title('Group Cumulative Return Curve (Equal Weight)', fontsize=14)
plt.ylabel('Cumulative Return', fontsize=12)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))  # Display as percentage
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('group_cumulative_return.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Plot group excess cumulative return curves
plt.figure(figsize=(12, 6))
for group in sorted(cumulative_excess_ret.columns):
    plt.plot(cumulative_excess_ret.index,
             cumulative_excess_ret[group],
             label=f'Group {group}',
             linewidth=1.5)

# Add zero line
plt.axhline(0, color='black', linestyle='--', linewidth=1)

plt.title('Group Excess Cumulative Return Curve (vs CSI 800)', fontsize=14)
plt.ylabel('Excess Cumulative Return', fontsize=12)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('group_excess_return.png', dpi=300, bbox_inches='tight')
plt.close()


print("Visualization charts saved as PNG files")