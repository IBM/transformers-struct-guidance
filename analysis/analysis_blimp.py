import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib import rcParams
from matplotlib.patches import Patch
rcParams['font.family'] = 'Arial'


def load_results(path):
    with open(path) as f:
        lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    lines = [(line[0], float(line[1])) for line in lines]
    return lines


phenomena = {
    "Anaphor Agreement": ["anaphor_gender_agreement", "anaphor_number_agreement"],
    "Argument Structure": ["animate_subject_passive", "animate_subject_trans", "causative", "drop_argument", "inchoative", "intransitive",
                            "passive_1", "passive_2", "transitive"],
    "Binding": ["principle_A_c_command", "principle_A_case_1", "principle_A_case_2", 
                "principle_A_domain_1", "principle_A_domain_2", "principle_A_domain_3", "principle_A_reconstruction"],
    "Control/Raising": ["existential_there_object_raising", "existential_there_subject_raising", "expletive_it_object_raising",
                        "tough_vs_raising_1", "tough_vs_raising_2"],
    "Determiner-Noun Agreement": ["determiner_noun_agreement_1", "determiner_noun_agreement_2", 
                                    "determiner_noun_agreement_irregular_1", "determiner_noun_agreement_irregular_2",
                                    "determiner_noun_agreement_with_adjective_1", "determiner_noun_agreement_with_adj_2",
                                    "determiner_noun_agreement_with_adj_irregular_1", "determiner_noun_agreement_with_adj_irregular_2"],
    "Ellipsis": ["ellipsis_n_bar_1", "ellipsis_n_bar_2"],
    "Filler Gap": ["wh_questions_object_gap", "wh_questions_subject_gap", "wh_questions_subject_gap_long_distance",
                    "wh_vs_that_no_gap", "wh_vs_that_no_gap_long_distance", "wh_vs_that_with_gap", "wh_vs_that_with_gap_long_distance"],
    "Irregular Forms": ["irregular_past_participle_adjectives", "irregular_past_participle_verbs"],
    "Island Effects": ["adjunct_island", "complex_NP_island", "coordinate_structure_constraint_complex_left_branch",
                        "coordinate_structure_constraint_object_extraction", "left_branch_island_echo_question",
                        "left_branch_island_simple_question", "sentential_subject_island", "wh_island"],
    "NPI Licensing": ["matrix_question_npi_licensor_present", "npi_present_1", "npi_present_2", "only_npi_licensor_present",
                        "only_npi_scope", "sentential_negation_npi_licensor_present", "sentential_negation_npi_scope"],
    "Quantifiers": ["existential_there_quantifiers_1", "existential_there_quantifiers_2", "superlative_quantifiers_1", "superlative_quantifiers_2"],
    "Subject-Verb Agreement": ["distractor_agreement_relational_noun", "distractor_agreement_relative_clause",
                                "irregular_plural_subject_verb_agreement_1", "irregular_plural_subject_verb_agreement_2",
                                "regular_plural_subject_verb_agreement_1", "regular_plural_subject_verb_agreement_2"],
}

corpora = ['bllip-md', 'bllip-lg']
models = ['xlm', 'sclm-past', 'sclm-next', 'xplm', 'xplm-mask']
model2seed = {'bllip-md':{'xlm':['1101', '1102', '1103'], 'sclm-past':['1101','1102', '1103', '1104'], 'sclm-next':['1101','1102', '1103','1104'], 'xplm':['1101', '1102', '1103'], 'xplm-mask':['1101', '1102', '1103']},
              'bllip-lg':{'xlm':['1101', '1102', '1103'], 'sclm-past':['1101', '1102', '1103', '1104'], 'sclm-next':['1101','1102','1103', '1104'], 'xplm':['1101', '1102', '1103'], 'xplm-mask':['1101', '1102', '1103']},}
model2name = {'xlm':'LM', 'sclm-past':'ScLM-past', 'sclm-next':'ScLM-next', 'xplm':'PLM', 'xplm-mask':'PLM-mask'}

rs = {}

for corpus in corpora:
    rs[corpus] = {}
    for model in models:
        rs[corpus][model] = {}

for corpus in corpora:
    for model in models:
        for seed in model2seed[corpus][model]:
            rs[corpus][model][seed] = {}
            path = 'results/blimp/{}_{}_rand-init_{}_5.txt'.format(model, corpus, seed)
            results = load_results(path)
            acc_all = [line[1] for line in results]
            acc_mean = np.mean(acc_all)
            rs[corpus][model][seed]['score'] = acc_mean
            rs[corpus][model][seed]['results'] = results


results = load_results('results/blimp/gpt-2.txt')
acc_all = [line[1] for line in results] 
acc_mean = np.mean(acc_all)
rs['gpt-2'] = {}
rs['gpt-2']['score'] = acc_mean
rs['gpt-2']['results'] = results


# Print the mean and standard deviation of model performances
for corpus in corpora:
    for model in models:
        rs_list = [rs[corpus][model][seed]['score'] for seed in model2seed[corpus][model]]
        print(corpus, model, 'mean:', np.mean(rs_list), 'std:', np.std(rs_list))


# Bar plot of model performances
model2style = {'xlm':{'color':plt.cm.Set2(7)}, 'sclm-past':{'color':plt.cm.Set2(0)}, 'sclm-next':{'color':plt.cm.Set2(1)},
                 'xplm':{'color':plt.cm.Set2(2)}, 'xplm-mask':{'color':plt.cm.Set2(3)}}
fig = plt.figure(figsize=(4, 2))
ax = plt.gca()
ax.grid(linestyle='--')
ax.set_axisbelow(True)
bar_width = 0.10

corpus2color = {'bllip-lg':plt.cm.tab20(0), 'bllip-md':plt.cm.tab20(1)}
for i, corpus in enumerate(corpora):
    for j, model in enumerate(models):
        acc_mean_list = []
        for seed in sorted(list(rs[corpus][model].keys())):
            acc_mean = rs[corpus][model][seed]['score']
            acc_mean_list.append(acc_mean)

        bar_position = i*1 + (bar_width+0.02)*(j-2)
        ax.bar(bar_position, np.mean(acc_mean_list), width=bar_width, **model2style[model], label=model)
        plt.plot([bar_position for _ in range(len(acc_mean_list))], acc_mean_list, 'o', linestyle='None', mec='k',color='k', mfc='None', markersize=5)


ax.bar(bar_position + bar_width*4, rs['gpt-2']['score'], width=bar_width, color='white', edgecolor='dimgrey',hatch='///', label='gpt-2')
ax.set_xlim(-0.5,len(corpora)-0.2)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_ylim((0.55, 0.83))
ax.set_yticks(np.arange(0.55, 0.82, 0.05))

ax.set_xticks(np.arange(0, len(corpora)))
ax.set_xticklabels([corpus.upper() for corpus in corpora], fontsize=10)
ax.set_ylabel('Accuracy', fontsize=10)

item_legend_elements = []
for model in models:
    patch = mpatches.Patch(**model2style[model], label=model2name[model])
    # item_legend_elements.append(Line2D([0], [0], **tag_cat2style[tag_cat],label='  '.join(tag_cat.split('_')), marker='o', linestyle = 'None', markersize=5))
    item_legend_elements.append(patch)

patch = Patch(facecolor='white', edgecolor='dimgrey',hatch='///', label='GPT-2')
item_legend_elements.append(patch)

ax.legend(handles=item_legend_elements, ncol=3, loc = 'center', bbox_to_anchor=(0.5, -0.36))

plt.title('Model Performance on BLiMP-10% Test Suites')
plt.savefig('figs/blimp_comparison_by_corpus.pdf'.format(corpus), bbox_inches="tight")
# plt.show()
plt.close()



# Bar plot of model performances on each cluster of phenomena
total_test_suite_count = 0
n_suite_dict = {}

for k, phenomenon in enumerate(list(phenomena.keys())):
    n_suite_dict[phenomenon] = 0
    tag_set = set(phenomena[phenomenon])
    for test_suite_name, _ in rs['bllip-md']['xlm']['1101']['results']:
        test_suite_cat = test_suite_name
        if test_suite_cat in tag_set:
            total_test_suite_count += 1
            n_suite_dict[phenomenon] += 1
assert total_test_suite_count == 67

fig, axes = plt.subplots(3, 4, sharex=True, figsize=(15,6))
plt.subplots_adjust(hspace = 0.3, wspace=0.2)
for k, phenomenon in enumerate(list(phenomena.keys())):
    tag_set = set(phenomena[phenomenon])
    
    ax = axes[k // 4, k % 4]

    ax.grid(linestyle='--')
    ax.set_axisbelow(True)
    bar_width = 0.10

    corpus2color = {'bllip-lg':plt.cm.tab20(0), 'bllip-md':plt.cm.tab20(1)}

    phenomenon_score_all = []

    for i, corpus in enumerate(corpora):
        for j, model in enumerate(models):
            acc_mean_list = []
            for seed in sorted(list(rs[corpus][model].keys())):
                score_list = []
                for test_suite_name, score in rs[corpus][model][seed]['results']:
                    test_suite_cat = test_suite_name
                    if test_suite_cat in tag_set:
                        score_list.append(score)
                acc_mean = np.mean(score_list)
                acc_mean_list.append(acc_mean)
                phenomenon_score_all.append(acc_mean)

            bar_position = i*1 + (bar_width+0.02)*(j-2)
            ax.bar(bar_position, np.mean(acc_mean_list), width=bar_width, **model2style[model], label=model)
            ax.plot([bar_position for _ in range(len(acc_mean_list))], acc_mean_list, 'o', linestyle='None', mec='k',color='k', mfc='None', markersize=5)

    ax.set_xlim(-0.5,len(corpora)-0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(np.arange(0, len(corpora)))
    ymin = np.max([(np.min(phenomenon_score_all) // 0.05 - 1)*0.05, 0])
    ymax = np.min([(np.max(phenomenon_score_all) // 0.05 + 1)*0.05, 1])
    ax.set_ylim((ymin, ymax))
    ax.set_yticks(np.arange(ymin, ymax, 0.05))
    ax.set_xticklabels([corpus.upper() for corpus in corpora], fontsize=10)
    if k % 4 == 0:
        ax.set_ylabel('Accuracy', fontsize=10)
    n_suite = n_suite_dict[phenomenon]
    if n_suite == 1:
        ax.set_title(phenomenon+' ({} suite)'.format(n_suite), fontsize=11.5)
    else:
        ax.set_title(phenomenon+' ({} suites)'.format(n_suite), fontsize=11.5)

    if k == 10:
        item_legend_elements = []
        for model in models:
            patch = mpatches.Patch(**model2style[model], label=model2name[model])
            item_legend_elements.append(patch)
        fig.legend(handles=item_legend_elements, ncol=5, loc = 'center', bbox_to_anchor=(0.5,0.03))    

fig.suptitle('Model Performance on Specific Clusters of BLiMP-10% Test Suites', x=0.5, y=0.97, fontsize=14)
plt.savefig('figs/blimp_comparison_by_test_suite_cluster_3_by_4.pdf', bbox_inches="tight")
# plt.show()
plt.close()


# Plot the model performance against word-level perplexity
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
plt.ylabel('BLiMP-10% Accuracy')

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

plt.savefig('figs/ppl_blimp.pdf', bbox_inches="tight")
# plt.show()
plt.close()
