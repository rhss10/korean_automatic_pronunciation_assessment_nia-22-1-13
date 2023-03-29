import matplotlib.pyplot as plt
import pandas as pd
import sys

input_log = sys.argv[1]  # recommended input file: data_stats.log as a result from data_stats.py
metric = "Fluency"
lang = sys.argv[2] # Selection: ASIA, CJ, ENG, EU

def read_log(input_log):
    lang_dict = {}
    for line in open(input_log, 'r').readlines()[1:]:
        lang = line.split(':',1)[0]
        dict_str = line.split(':',1)[1].strip() # get dictionary format part
        prof_cnt_dict = eval(dict_str.replace("'", "\"")) # covnert string to dict
        lang_dict[lang] = prof_cnt_dict
    return lang_dict

def convert_to_pd(lang, lang_dict):
    prof_cnt_dict = lang_dict[lang]
    # df = pd.DataFrame({'Proficiency':['1', '2', '3', '4', '5'],
    #                     'Counts':[prof_cnt_dict['1'], prof_cnt_dict['2'], prof_cnt_dict['3'], prof_cnt_dict['4'], prof_cnt_dict['5']]
    #                     })
    df = pd.DataFrame({'Proficiency':['0', '1', '2', '3', '4', '5'],
                        'Counts':[prof_cnt_dict['0'], prof_cnt_dict['1'], prof_cnt_dict['2'], prof_cnt_dict['3'], prof_cnt_dict['4'], prof_cnt_dict['5']]
                        })

    total = df.Counts.sum()
    percentage = []
    for i in range(df.shape[0]):
        pct = (df.Counts[i] / total) * 100
        percentage.append(round(pct, 2))
    df['Percentage'] = percentage

    return df

def plot_bar_with_percentage(df):
    plt.figure(figsize=(12,12))
    # colors_list = ['Red', 'Orange', 'Blue', 'Purple', 'Green']
    # colors_list = ['#ffa600', '#ff6361', '#bc5090', '#58508d', '#003f5c']
    colors_list = ['black', '#ffa600', '#ff6361', '#bc5090', '#58508d', '#003f5c']  # For 6 classes
    graph = plt.bar(df.Proficiency, df.Counts, color=colors_list)
    plt.xlabel('{} Level'.format(metric), fontsize=14)
    plt.ylabel('Num. of Samples', fontsize=14)
    plt.title("Counts and Percentage of {} Levels of {} data\n".format(metric, lang), fontsize=16)
    i = 0
    for p in graph:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(x+width/2,
                 y+height*1.01,
                 # str(df.Percentage[i])+'%\n'+str(df.Counts[i]),
                 str(df.Counts[i]) +'\n('+str(df.Percentage[i])+'%)',
                 ha='center',
                 weight='bold')
        i+=1
    plt.savefig('{}_{}_0to5_graph.png'.format(metric, lang))

def main():
    lang_dict = read_log(input_log)
    df = convert_to_pd(lang, lang_dict)
    plot_bar_with_percentage(df)

if __name__ == '__main__':
    main()
