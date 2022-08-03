import plotly.express as px
import pandas as pd

def bar_plot(results):
    df_DF_results = results['DF']['accuracy_test'].melt()
    df_DF_results['model'] = 'DF'
    df_DNN_results = results['DNN']['accuracy_test'].melt()
    df_DNN_results['model'] = 'DNN'
    df_RF_results = results['RF']['accuracy_test'].melt()
    df_RF_results['model'] = 'RF'
    df_DT_results = results['DT']['accuracy_test'].melt()
    df_DT_results['model'] = 'DT'
    df_SVM_results = results['SVM']['accuracy_test'].melt()
    df_SVM_results['model'] = 'SVM'
    all_results = [df_DF_results, df_DNN_results, df_RF_results, df_DT_results, df_SVM_results]
    df_all_results = pd.concat(all_results)
    df_all_results['variable'] = df_all_results['variable'].apply(lambda x: x * 100 + 100)
    fig = px.box(df_all_results, x='variable', y='value', color='model',
                 width=1500)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_yaxes(title='Accuracy Test Set')
    fig.update_xaxes(title='Size Sample', dtick=100, )
    fig.add_vrect(x0=150, x1=150, line_color='white', line_width=3)
    fig.add_vrect(x0=250, x1=250, line_color='white', line_width=3)
    fig.add_vrect(x0=350, x1=350, line_color='white', line_width=3)
    fig.add_vrect(x0=450, x1=450, line_color='white', line_width=3)
    fig.add_vrect(x0=550, x1=550, line_color='white', line_width=3)
    fig.add_vrect(x0=650, x1=650, line_color='white', line_width=3)
    fig.add_vrect(x0=750, x1=750, line_color='white', line_width=3)
    fig.add_vrect(x0=850, x1=850, line_color='white', line_width=3)
    fig.add_vrect(x0=950, x1=950, line_color='white', line_width=3)
    fig.update_layout(
        xaxis=dict(
            tickfont=dict(size=20),
            titlefont=dict(size=20)),
        yaxis=dict(
            tickfont=dict(size=15),
            titlefont=dict(size=20)))
    fig.show()
