import seaborn 
import numpy as np
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
import json 

def get_pretraining_data_delay():
    # Read delay data from the pre-training data 

    path = "/local/home/sidray/packet_transformer/evaluations/congestion_1/"

    files = ["endtoenddelay500s_1.csv", "endtoenddelay500s_2.csv",
             "endtoenddelay500s_3.csv", "endtoenddelay500s_4.csv",
            "endtoenddelay500s_5.csv"]

    list_of_delay_lists = []
    list_of_tuples = []

    for count,file in enumerate(files):
        df_pretraining = pd.read_csv(path+file)
        delay_list = (df_pretraining["Delay"]*1000).to_list()
        list_of_delay_lists.append(delay_list)
        delay_arr = np.array(delay_list)
        metric_tuple = (np.mean(delay_arr), np.median(delay_arr), np.quantile(delay_arr, 0.99))
        list_of_tuples.append(metric_tuple)

    return list_of_delay_lists, list_of_tuples

def get_memento_finetuning_data_delay():

    path = "/local/home/sidray/packet_transformer/evaluations/memento_data/"

    files = ["memento_test10_final.csv"]

    list_of_delay_lists = []
    list_of_tuples = []

    for count,file in enumerate(files):
        df_pretraining = pd.read_csv(path+file)
        delay_list = (df_pretraining["Delay"]*1000).to_list()
        list_of_delay_lists.append(delay_list)
        delay_arr = np.array(delay_list)
        metric_tuple = (np.mean(delay_arr), np.median(delay_arr), np.quantile(delay_arr, 0.99))
        list_of_tuples.append(metric_tuple)

    return list_of_delay_lists, list_of_tuples 


def get_memento_finetuning_data_bigger_topology():

    path = "/local/home/sidray/packet_transformer/evaluations/memento_data/"

    files = ["topo_1_final.csv", "topo_2_final.csv", "topo_test_1_final.csv"]

    list_of_delay_lists = []
    list_of_tuples = []

    for count,file in enumerate(files):
        df_pretraining = pd.read_csv(path+file)
        delay_list = (df_pretraining["Delay"]*1000).to_list()
        list_of_delay_lists.append(delay_list)
        delay_arr = np.array(delay_list)
        metric_tuple = (np.mean(delay_arr), np.median(delay_arr), np.quantile(delay_arr, 0.99))
        list_of_tuples.append(metric_tuple)

    return list_of_delay_lists, list_of_tuples 



def plot_delay_cdf(delay_arrray, name="Experiment"):
    path = "plots/"
    delay_list = np.array(delay_arrray)
    kwargs = {'cumulative': True}
    plt.figure()
    sbs = sns.distplot(delay_arrray, hist_kws=kwargs, kde_kws=kwargs)
    sbs.set_title(name)
    sbs.set_xlabel('Delay')
    # fig = sbs.get_figure()
    plt.savefig(path+name+".png")


if __name__=="__main__":


    pretrain_delay, pretrained_tuples = get_pretraining_data_delay()
    memento_finetune_delay_simple, memento_simple_tuples = get_memento_finetuning_data_delay()
    memento_finetune_delay_complex, memento_complex_tuples = get_memento_finetuning_data_bigger_topology()

    for count, value in enumerate(pretrain_delay):
        plot_delay_cdf(value, "Pretraining-Data-Delay-Seed{}".format(count))

    plot_delay_cdf(memento_finetune_delay_simple, "Memento-Finetuning-SimpleTopo")

    for count, value in enumerate(memento_finetune_delay_complex):
        plot_delay_cdf(value, "Memento-Finetuning-ScaledupTopo_{}".format(count+1))

    global_metrics_dict = {}
    global_metrics_dict["Pretrained"] = {}
    global_metrics_dict["Memento-Finetune-Base"] = {}
    global_metrics_dict["Memento-Finetune-Scaled"] = {}

    for count, value in enumerate(pretrained_tuples):
        global_metrics_dict["Pretrained"]["Seed_{}".format(count)] = (("Mean", value[0]),
                                                                     ("Median", value[1]), ("99ile", value[2]))

    global_metrics_dict["Memento-Finetune-Base"]["Base_value"] =  (("Mean", memento_simple_tuples[0][0]), 
                                                                    ("Median", memento_simple_tuples[0][1]), ("99ile", memento_simple_tuples[0][2]))

    for count, value in enumerate(memento_complex_tuples):
        global_metrics_dict["Memento-Finetune-Scaled"]["Topo_{}".format(count+1)] = (("Mean", value[0]),
                                                                     ("Median", value[1]), ("99ile", value[2]))

    print(global_metrics_dict)   

    out_json = json.dumps(global_metrics_dict, indent=4) 

    with open('plots/metrics_json_data.json', 'w') as outfile:
        outfile.write(out_json)                                            
    