import numpy as np
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'


def load_results(path):
    with open(path) as f:
        lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    lines = [(line[0], float(line[1])) for line in lines]
    return lines


circuits = {
    "Licensing": ["npi", "reflexive"],
    "Long-Distance Dependencies": ["fgd", "cleft"],
    "Agreement": ["number"],
    "Garden-Path Effects": ["npz", "mvrr"],
    "Gross Syntactic State": ["subordination"],
    "Center Embedding": ["center"],
}

corpora = ['bllip-md', 'bllip-lg']
models = ['xlm', 'sclm-past', 'sclm-next', 'xplm', 'xplm-mask']
model2seed = {'bllip-md':{'xlm':['1101', '1102', '1103'], 'sclm-past':['1101', '1102', '1103', '1104'], 'sclm-next':['1101', '1102', '1103', '1104'], 'xplm':['1101', '1102', '1103'], 'xplm-mask':['1101', '1102', '1103']},
              'bllip-lg':{'xlm':['1101', '1102', '1103'], 'sclm-past':['1101', '1102', '1103', '1104'], 'sclm-next':['1101', '1102', '1103', '1104'], 'xplm':['1101', '1102', '1103'], 'xplm-mask':['1101', '1102', '1103']},}

model2name = {'xlm':'LM', 'sclm-past':'ScLM-past', 'sclm-next':'ScLM-next', 'xplm':'PLM', 'xplm-mask':'PLM-mask', 'rnng':'RNNG'}

rs = {}

for corpus in corpora:
    rs[corpus] = {}
    for model in models:
        rs[corpus][model] = {}

for corpus in corpora:
    for model in models:
        for seed in model2seed[corpus][model]:
            rs[corpus][model][seed] = {}
            path = 'results/sg/{}_{}_rand-init_{}_5.txt'.format(model, corpus, seed)
            results = load_results(path)
            acc_all = [line[1] for line in results if line[0] != 'nn-nv-rpl'] # filter one test suite
            acc_mean = np.mean(acc_all)
            rs[corpus][model][seed]['score'] = acc_mean
            rs[corpus][model][seed]['results'] = [line for line in results if line[0] != 'nn-nv-rpl'] # filter one test suite


results = load_results('results/sg/gpt-2.txt')
acc_all = [line[1] for line in results if line[0] != 'nn-nv-rpl'] # filter one test suite
acc_mean = np.mean(acc_all)
rs['gpt-2'] = {}
rs['gpt-2']['score'] = acc_mean
rs['gpt-2']['results'] = [line for line in results if line[0] != 'nn-nv-rpl'] # filter one test suite


# Print the mean and standard deviation of model performances
for corpus in corpora:
    for model in models:
        rs_list = [rs[corpus][model][seed]['score'] for seed in model2seed[corpus][model]]
        print(corpus, model, 'mean:', np.mean(rs_list), 'std:', np.std(rs_list))


# Plot the SG score against word-level perplexity
with open('results/model_ppl_list.txt') as f:
    lines = f.readlines()
lines = [line.strip().split() for line in lines if line.strip() != '']
ppl_results = [(line[0], line[1], line[2], float(line[3])) for line in lines]

model2style = {'xlm':{'marker':'o', 'ms':5}, 'xplm':{'marker':'x', 'ms':5}, 'xplm-mask':{'marker':'^', 'ms':5}, 
                'sclm-past':{'marker':'s', 'ms':5}, 'sclm-next':{'marker':'d', 'ms':5},}
corpus2style = {'bllip-lg':{'color':plt.cm.tab20(0)}, 'bllip-md':{'color':plt.cm.tab20(1)}}

fig = plt.figure(figsize=(2.5,2.5))

for model, corpus, seed, ppl in ppl_results:
    if seed not in rs[corpus][model]:
        print(model, corpus, seed)
    plt.plot(ppl, rs[corpus][model][seed]['score'],  **model2style[model], **corpus2style[corpus])
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlabel('Word-level Perplexity')
plt.ylabel('SG Accuracy')

item_legend_elements = []
for model in models:
    item_legend_elements.append(Line2D([0], [0], **model2style[model],label=model2name[model], color='gray', linestyle = 'None'))
legend = plt.legend(handles=item_legend_elements, loc = 'center left', bbox_to_anchor=(1.1, 0.25), title='Model')
ax.add_artist(legend)

item_legend_elements = []
for corpus in corpora:
    patch = mpatches.Patch(**corpus2style[corpus], label=corpus.upper())
    item_legend_elements.append(patch)
legend = plt.legend(handles=item_legend_elements, loc = 'center left', bbox_to_anchor=(1.1, 0.8), title='Corpus')
ax.add_artist(legend)

plt.savefig('figs/ppl_sg.pdf', bbox_inches="tight")
# plt.show()
plt.close()


# Load RNNG results
rnng_rs = {}

rnng2seed = {'bllip-md':['44862', '3602'], 'bllip-lg':['38435', '62488', '7245']}

for corpus in corpora:
    rnng_rs[corpus] = {}
    path = 'results/sg/syntaxgym_results_rnng_{}.csv'.format(corpus)
    df = pd.read_csv(path)

    test_suite_name_list = list(set(df['suite'].tolist()))

    for seed in rnng2seed[corpus]:
        rnng_rs[corpus][seed] = {}
        acc_list = []
        for test_suite_name in test_suite_name_list:
            data = df[df['model'].isin(['rnng_{}_{}'.format(corpus, seed)]) & df['suite'].isin([test_suite_name])]['correct'].tolist()
            acc_list.append([test_suite_name, np.mean(data)])
        rnng_rs[corpus][seed]['results'] = acc_list
        rnng_rs[corpus][seed]['score'] = np.mean([acc for _, acc in acc_list])


# Bar plot of model performances
model2style = {'xlm':{'color':plt.cm.Set2(7)}, 'sclm-past':{'color':plt.cm.Set2(0)}, 'sclm-next':{'color':plt.cm.Set2(1)},
                 'xplm':{'color':plt.cm.Set2(2)}, 'xplm-mask':{'color':plt.cm.Set2(3)}, 'rnng':{'color':plt.cm.Set2(4)},}
fig = plt.figure(figsize=(4, 2))
ax = plt.gca()
ax.grid(linestyle='--')
ax.set_axisbelow(True)
bar_width = 0.10

corpus2color = {'bllip-lg':plt.cm.tab20(0), 'bllip-md':plt.cm.tab20(1)}
for i, corpus in enumerate(corpora):
    for j, model in enumerate(['rnng']+models):
        acc_mean_list = []

        if model == 'rnng':
            for seed in rnng2seed[corpus]:
                acc_mean = rnng_rs[corpus][seed]['score']
                acc_mean_list.append(acc_mean)
        else:                
            for seed in sorted(list(rs[corpus][model].keys())):
                acc_mean = rs[corpus][model][seed]['score']
                acc_mean_list.append(acc_mean)

        bar_position = i*1 + (bar_width+0.02)*(j-2.5)

        if model == 'rnng':
            ax.bar(bar_position, np.mean(acc_mean_list), width=bar_width, color=plt.cm.Set2(4), label=model)
        else:
            ax.bar(bar_position, np.mean(acc_mean_list), width=bar_width, **model2style[model], label=model)
        plt.plot([bar_position for _ in range(len(acc_mean_list))], acc_mean_list, 'o', linestyle='None', mec='k',color='k', mfc='None', markersize=5)

ax.bar(bar_position + bar_width*4, rs['gpt-2']['score'], width=bar_width, color='white', edgecolor='dimgrey',hatch='///', label='gpt-2')

ax.set_xlim(-0.5,len(corpora)-0.2)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_ylim((0.55, 0.82))
ax.set_yticks(np.arange(0.55, 0.82, 0.05))
ax.set_xticks(np.arange(0, len(corpora)))
ax.set_xticklabels([corpus.upper() for corpus in corpora], fontsize=10)
ax.set_ylabel('Accuracy', fontsize=10)

item_legend_elements = []
for model in ['rnng'] + models:
    patch = mpatches.Patch(**model2style[model], label=model2name[model])
    item_legend_elements.append(patch)

patch = Patch(facecolor='white', edgecolor='dimgrey',hatch='///', label='GPT-2')

item_legend_elements.append(patch)

ax.legend(handles=item_legend_elements, ncol=3, loc = 'center', bbox_to_anchor=(0.5, -0.36))

plt.title('Model Performance on SG Test Suites')
plt.savefig('figs/sg_comparison_by_corpus_with_rnng.pdf'.format(corpus), bbox_inches="tight")
# plt.show()
plt.close()


# Bar plot of model performances on test suite clusters
total_test_suite_count = 0

n_suite_dict = {}

for k, circuit in enumerate(list(circuits.keys())):
    n_suite_dict[circuit] = 0
    tag_set = set(circuits[circuit])
    for test_suite_name, _ in rs['bllip-md']['xlm']['1101']['results']:
        test_suite_cat = re.split('_|-',test_suite_name)
        if test_suite_cat[0] in tag_set:
            total_test_suite_count += 1
            n_suite_dict[circuit] += 1
assert total_test_suite_count == 33

fig, axes = plt.subplots(2, 4, sharex=True, figsize=(15, 4))
plt.subplots_adjust(hspace = 0.3, wspace=0.2)
for k, circuit in enumerate(list(circuits.keys())):
    tag_set = set(circuits[circuit])
    
    ax = axes[k // 3, k % 3]

    ax.grid(linestyle='--')
    ax.set_axisbelow(True)
    bar_width = 0.10

    corpus2color = {'bllip-lg':plt.cm.tab20(0), 'bllip-md':plt.cm.tab20(1)}

    circuit_score_all = []

    for i, corpus in enumerate(corpora):
        for j, model in enumerate(['rnng']+models):
            acc_mean_list = []
            if model == 'rnng':
                seeds = rnng2seed[corpus]
            else:
                seeds = sorted(list(rs[corpus][model].keys()))
            for seed in seeds:
                score_list = []
                if model == 'rnng':
                    results_by_suites = rnng_rs[corpus][seed]['results']
                else:
                    results_by_suites = rs[corpus][model][seed]['results']
                for test_suite_name, score in results_by_suites:
                    test_suite_cat = re.split('_|-',test_suite_name)
                    if test_suite_cat[0] in tag_set:
                        score_list.append(score)
                acc_mean = np.mean(score_list)
                acc_mean_list.append(acc_mean)
                circuit_score_all.append(acc_mean)

            bar_position = i*1 + (bar_width+0.02)*(j-2)
            ax.bar(bar_position, np.mean(acc_mean_list), width=bar_width, **model2style[model], label=model)
            ax.plot([bar_position for _ in range(len(acc_mean_list))], acc_mean_list, 'o', linestyle='None', mec='k',color='k', mfc='None', markersize=5)

    ax.set_xlim(-0.5,len(corpora)-0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(np.arange(0, len(corpora)))
    ymin = (np.min(circuit_score_all) // 0.05 - 1)*0.05
    ymax = np.min([(np.max(circuit_score_all) // 0.05 + 1)*0.05, 1])
    ax.set_ylim((ymin, ymax))

    ax.set_yticks(np.arange(ymin, ymax, 0.05))

    n_yticks = (ymax - ymin)/0.05
    if n_yticks > 8:
        yticklabels = []
        for j, y_value in enumerate(np.arange(ymin, ymax, 0.05)):
            if j % 2 == 0:
                yticklabels.append(np.round(y_value,2))
            else:
                yticklabels.append('')
        ax.set_yticklabels(yticklabels)
    ax.set_xticklabels([corpus.upper() for corpus in corpora], fontsize=10)
    if k % 3 == 0:
        ax.set_ylabel('Accuracy', fontsize=10)
    
    n_suite = n_suite_dict[circuit]
    if n_suite == 1:
        ax.set_title(circuit+' ({} suite)'.format(n_suite), fontsize=11.5)
    else:
        ax.set_title(circuit+' ({} suites)'.format(n_suite), fontsize=11.5)

    if k == 4:
        item_legend_elements = []
        for model in ['rnng']+models:
            patch = mpatches.Patch(**model2style[model], label=model2name[model])
            item_legend_elements.append(patch)
        fig.legend(handles=item_legend_elements, ncol=1, loc = 'center left', bbox_to_anchor=(0.72, 0.75))   

axes[0, -1].axis('off')
axes[1, -1].axis('off')

fig.suptitle('Model Performance on Specific Clusters of SG Test Suites', x=0.5, y=1, fontsize=14)

plt.savefig('figs/sg_comparison_by_test_suite_cluster_2_by_4_with_rnng.pdf', bbox_inches="tight")
# plt.show()
plt.close()