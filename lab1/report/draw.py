"""Audit existing results and regenerate publication figures. No model training.

Run: py -3.13 draw.py (or python draw.py with the requirements installed).
All paths are relative to this file, so the working directory is immaterial.
"""
from pathlib import Path
import ast
import contextlib
import hashlib
import io
import json
import math
import re
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
LABELS = ['business', 'politics', 'sports']
METHODS = ['Binary BoW', 'Frequency BoW', 'NYT W2V', 'AG W2V', 'GloVe', 'BERT']
STEMS = ['binary_bow', 'frequency_bow', 'nyt_word2vec', 'ag_word2vec', 'glove_100d', 'bert_finetune']
TASKS = [1, 1, 2, 2, 2, 3]
COLORS = ['#264653', '#4e8790', '#2878a0', '#78a6bd', '#d7a04b', '#b95c50']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.axisbelow': True, 'savefig.dpi': 180})
AUDIT = {'checks': [], 'files': [], 'limitations': [
    'No training, checkpoint inference or embedding retraining was performed.',
    'OOV numerators are checked against saved counts; embedding files are absent, so membership cannot be independently recomputed.',
    'Validation predictions are absent; validation Macro-F1 can only be cross-checked across saved summaries.',
    'BERT length 64 includes special tokens; no exact WordPiece truncation rate is inferred from word counts.',
    'Single seed/run; numerical ranking is not evidence of statistical significance.'
]}

def check(condition, description):
    if not condition:
        raise AssertionError(description)
    AUDIT['checks'].append(description)

def close(a, b, description):
    check(np.allclose(a, b, atol=1e-12, rtol=0), description)

def save(fig, name):
    fig.savefig(FIG / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(FIG / (name + '.png'), bbox_inches='tight')
    plt.close(fig)

def heatmap(ax, values, cmap, vmax, aspect='equal'):
    """Draw discrete matrix cells as vector polygons, not a raster image."""
    rows,cols = values.shape
    ax.pcolormesh(np.arange(cols+1)-.5, np.arange(rows+1)-.5, values,
        cmap=cmap, vmin=0, vmax=vmax, shading='flat', rasterized=False)
    ax.set_xlim(-.5,cols-.5)
    ax.set_ylim(rows-.5,-.5)
    ax.set_aspect(aspect)

def selected_functions(script, names, namespace):
    tree = ast.parse(script.read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(script), 'exec'), namespace)

def audit():
    raw = pd.read_csv(ROOT / 'nyt.csv')
    data = raw[['text', 'label']].dropna().copy()
    data['text'] = data.text.astype(str)
    data['label'] = data.label.astype(str)
    data = data[data.text.str.strip() != '']
    check(set(data.label) == set(LABELS), 'Exactly three NYT labels')
    splits = []
    for task in [1, 2, 3]:
        ns = {'pd': pd, 'train_test_split': train_test_split, 'RANDOM_STATE': 42,
              'print_split_distribution': lambda *args: None}
        script = ROOT / ['task1_bow.py', 'task2_word2vec.py', 'task3_bert.py'][task-1]
        selected_functions(script, ['split_data', 'split_nyt', 'print_split_distribution'], ns)
        with contextlib.redirect_stdout(io.StringIO()):
            splits.append(ns['split_nyt' if task == 2 else 'split_data'](data.copy()))
    for other in splits[1:]:
        check(all(list(a) == list(b) for a, b in zip(splits[0], other)), 'Actual script split functions agree in text and label order')
    xt, xv, xe, yt, yv, ye = splits[0]
    check(len(xt)+len(xv)+len(xe) == len(data), 'Split sizes sum to cleaned corpus')
    check(not(set(xt.index) & set(xv.index) or set(xt.index) & set(xe.index) or set(xv.index) & set(xe.index)), 'Split row indices are disjoint')
    AUDIT['data'] = {'raw': len(raw), 'clean': len(data), 'removed': len(raw)-len(data),
        'counts': data.label.value_counts().to_dict(), 'splits': {},
        'duplicate_texts': int(data.text.duplicated().sum()),
        'cross_split_duplicate_texts': {'train_val': len(set(xt)&set(xv)), 'train_test': len(set(xt)&set(xe)), 'val_test': len(set(xv)&set(xe))}}
    for name, x, y in [('train',xt,yt),('validation',xv,yv),('test',xe,ye)]:
        AUDIT['data']['splits'][name] = {'n': len(x), 'counts': y.value_counts().to_dict()}
    ag = pd.read_csv(ROOT / 'ag.csv')
    AUDIT['data']['ag'] = {'rows': len(ag), 'columns': list(ag), 'nonempty': int((ag.iloc[:,0].fillna('').astype(str).str.strip()!='').sum())}
    AUDIT['data']['word2vec_corpora'] = {}
    for name,texts in [('nyt_train',xt),('ag',ag.text.dropna())]:
        counter = Counter()
        for text in texts:
            counter.update(re.findall(r'\b[a-z]+\b', str(text).lower()))
        AUDIT['data']['word2vec_corpora'][name] = {'tokens':sum(counter.values()),
            'vocabulary_min_count_2':sum(count>=2 for count in counter.values())}
    vocab = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w\w+\b').fit(xt)
    AUDIT['data']['bow_vocabulary'] = len(vocab.vocabulary_)
    summary = pd.read_csv(ROOT / 'task3_results/all_tasks_comparison.csv')
    check(list(summary.Representation) == ['Binary Bag of Words','Frequency Bag of Words','nyt_word2vec','ag_word2vec','glove_100d','BERT-base-uncased Fine-tuning'], 'Combined summary method order')
    predictions, matrices, reports = [], [], []
    check(not pd.Series(list(xe)).duplicated().any(), 'Test texts unique: exact-text reconstruction is unambiguous')
    for i, (stem, task) in enumerate(zip(STEMS,TASKS)):
        directory = ROOT / f'task{task}_results'
        wrong = pd.read_csv(directory / f'{stem}_wrong_cases.csv')
        predcol = 'bert_pred' if task == 3 else 'pred_label'
        lookup = dict(zip(wrong.text,wrong[predcol]))
        check(len(lookup)==len(wrong), f'{stem}: wrong-case texts unique')
        truth_lookup = dict(zip(xe,ye))
        check(all(text in truth_lookup and truth_lookup[text]==label for text,label in zip(wrong.text,wrong.true_label)), f'{stem}: all wrong cases belong to exact test split with correct labels')
        check(all(wrong.true_label != wrong[predcol]), f'{stem}: saved errors are genuine errors')
        pred = [lookup.get(text,label) for text,label in zip(xe,ye)]
        cr = classification_report(ye,pred,labels=LABELS,output_dict=True)
        saved = pd.read_csv(directory / f'{stem}_classification_report.csv',index_col=0)
        for label in LABELS + ['macro avg','weighted avg']:
            for metric in ['precision','recall','f1-score','support']:
                close(cr[label][metric], saved.loc[label,metric], f'{stem}: {label} {metric}')
        close(cr['accuracy'], saved.loc['accuracy','precision'], f'{stem}: saved accuracy row')
        acc, f1 = accuracy_score(ye,pred), f1_score(ye,pred,average='macro')
        close(acc,summary.iloc[i]['Test Accuracy'], f'{stem}: recomputed test Accuracy matches combined summary')
        close(f1,summary.iloc[i]['Test Macro-F1'], f'{stem}: recomputed test Macro-F1 matches combined summary')
        own = pd.read_csv(directory / f'task{task}_summary.csv')
        row = own.iloc[i if task == 1 else i-2 if task == 2 else 0]
        close([acc,f1],[row['Test Accuracy'],row['Test Macro-F1']],f'{stem}: task summary matches')
        if task == 1:
            check(row['Vocabulary Size']==len(vocab.vocabulary_), f'{stem}: vocabulary independently recomputed')
        elif task == 2:
            check(row['Dimension']==100, f'{stem}: embedding dimension')
        if 'Validation Accuracy' in row:
            close(row['Validation Accuracy']*len(xv),round(row['Validation Accuracy']*len(xv)),f'{stem}: validation Accuracy is consistent with integer correct count')
        cm = confusion_matrix(ye,pred,labels=LABELS)
        pd.DataFrame(cm,index=LABELS,columns=LABELS).to_csv(FIG / f'{stem}_confusion_matrix.csv')
        predictions.append(pred); matrices.append(cm); reports.append(cr)
    # Exact checks of all saved disagreement/subset CSVs, including both-wrong cases.
    pairs = [(0,1,1,'binary_vs_frequency_disagreement','binary_pred','frequency_pred'),
             (2,3,2,'nyt_vs_ag_word2vec_disagreement','nyt_word2vec_pred','ag_word2vec_pred')]
    for a,b,task,stem,ca,cb in pairs:
        pa,pb = predictions[a],predictions[b]
        expected = [(t,y,u,v) for t,y,u,v in zip(xe,ye,pa,pb) if u!=v]
        saved = pd.read_csv(ROOT / f'task{task}_results/{stem}.csv')
        check(list(saved[['text','true_label',ca,cb]].itertuples(index=False,name=None))==expected, f'{stem}: exact saved disagreement rows')
        subset_stems = ['binary_right_frequency_wrong','frequency_right_binary_wrong'] if task==1 else ['nyt_right_ag_wrong','ag_right_nyt_wrong']
        for k,subset in enumerate(subset_stems):
            expected_subset = [r for r in expected if r[2]==r[1] and r[3]!=r[1]] if k==0 else [r for r in expected if r[3]==r[1] and r[2]!=r[1]]
            got = pd.read_csv(ROOT / f'task{task}_results/{subset}.csv')
            check(list(got[['text','true_label',ca,cb]].itertuples(index=False,name=None))==expected_subset, f'{subset}: exact correctness subset')
        AUDIT[stem] = {'total':len(expected),'a_right_b_wrong':sum(u==y and v!=y for _,y,u,v in expected),
            'b_right_a_wrong':sum(v==y and u!=y for _,y,u,v in expected),'both_wrong':sum(u!=y and v!=y for _,y,u,v in expected)}
    tokens = [re.findall(r'\b[a-z]+\b',str(t).lower()) for t in xe]
    total = sum(map(len,tokens)); unique = len(set(w for ts in tokens for w in ts))
    oov = pd.read_csv(ROOT / 'task2_results/oov_statistics.csv')
    task2 = pd.read_csv(ROOT / 'task2_results/task2_summary.csv')
    for i,row in oov.iterrows():
        check(row['Total Tokens']==total and row['Unique Words']==unique, f"{row['Representation']}: OOV denominators independently recomputed")
        close(row['OOV Tokens']/total,row['Token OOV Rate'],f"{row['Representation']}: token OOV arithmetic")
        close(row['Unique OOV Words']/unique,row['Unique OOV Rate'],f"{row['Representation']}: type OOV arithmetic")
        close([row['Token OOV Rate'],row['Unique OOV Rate']],task2.iloc[i][['Token OOV Rate','Unique Word OOV Rate']].to_numpy(dtype=float),f"{row['Representation']}: OOV summaries agree")
    history = pd.read_csv(ROOT / 'task3_results/training_history.csv')
    bert = pd.read_csv(ROOT / 'task3_results/task3_summary.csv').iloc[0]
    best = history.loc[history['Validation Macro-F1'].idxmax()]
    check(best.Epoch==bert['Best Epoch'], 'BERT best checkpoint epoch agrees with validation history')
    close(best['Validation Macro-F1'],bert['Best Validation Macro-F1'],'BERT best validation score')
    check(bert['Max Length']==64 and bert.Epochs==3 and bert['Batch Size']==8 and bert['Learning Rate']==2e-5,'BERT persisted hyperparameters')
    check(list(history.Epoch)==[1,2,3], 'Exactly three completed BERT epochs')
    AUDIT['bert_epoch_seconds'] = float(history['Epoch Time Seconds'].sum())
    check(json.loads((ROOT/'task3_results/label_mapping.json').read_text())['label2id']==dict(zip(LABELS,range(3))),'BERT label mapping')
    for stem in STEMS[:2]:
        words = pd.read_csv(ROOT / f'task1_results/{stem}_top_words.csv')
        for label,g in words.groupby('class'):
            check(label in LABELS and list(g['rank'])==list(range(1,21)) and g.coefficient.is_monotonic_decreasing, f'{stem}: {label} top-20 ranks and descending coefficients')
            check(set(g.word)<=set(vocab.vocabulary_),f'{stem}: {label} words belong to training vocabulary')
    for stem in STEMS[2:4]:
        words = pd.read_csv(ROOT / f'task2_results/{stem}_similar_words.csv')
        for query,g in words.groupby('query'):
            check(list(g['rank'])==list(range(1,11)) and g.similarity.is_monotonic_decreasing and g.similarity.between(-1,1).all(), f'{stem}: {query} neighbor ranks and cosine range')
    for p in sorted(ROOT.glob('task*_results/*')):
        if p.is_file():
            item = {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
            if p.suffix=='.csv':
                frame = pd.read_csv(p); item.update(rows=len(frame),columns=list(frame.columns))
                check(not frame.isna().any().any(), f'{p.name}: no missing CSV cells')
            AUDIT['files'].append(item)
    AUDIT['metrics'] = {m:{'accuracy':accuracy_score(ye,p),'macro_f1':f1_score(ye,p,average='macro'),'errors':int((np.array(p)!=np.array(ye)).sum()),'confusion_matrix':cm.tolist()} for m,p,cm in zip(METHODS,predictions,matrices)}
    train_texts = set(xt)
    overlap = np.array([t in train_texts for t in xe])
    AUDIT['train_test_overlap_diagnostic'] = {
        'overlap_n': int(overlap.sum()), 'remaining_n': int((~overlap).sum()),
        'methods': {m:{'overlap_correct':int(np.sum((np.array(p)==np.array(ye)) & overlap)),
            'remaining_accuracy':accuracy_score(np.array(ye)[~overlap],np.array(p)[~overlap]),
            'remaining_macro_f1':f1_score(np.array(ye)[~overlap],np.array(p)[~overlap],average='macro')}
            for m,p in zip(METHODS,predictions)}
    }
    AUDIT['source_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [ROOT/'nyt.csv',ROOT/'ag.csv',ROOT/'task1_bow.py',ROOT/'task2_word2vec.py',ROOT/'task3_bert.py'] + list(ROOT.glob('*.md'))}
    AUDIT['word_lengths'] = {'median':float(np.median([len(t) for t in tokens])), 'min':min(map(len,tokens)), 'max':max(map(len,tokens))}
    display_methods = ['Binary BoW','Frequency BoW','NYT Word2Vec','AG Word2Vec','GloVe','BERT']
    class_rows = [m + ' & ' + ' & '.join(f"{cr[label]['f1-score']*100:.2f}" for label in LABELS) + r'\\' for m,cr in zip(display_methods,reports)]
    (HERE/'class_table.tex').write_text('% Auto-generated by draw.py from audited CSVs.\n' +
        r'\begin{tabular}{lrrr}' + '\n' + r'\toprule' + '\n' +
        r'方法 & 商业 F1 & 政治 F1 & 体育 F1\\' + '\n' + r'\midrule' + '\n' +
        '\n'.join(class_rows) + '\n' + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n',encoding='utf-8')
    (HERE/'audit.json').write_text(json.dumps(AUDIT,ensure_ascii=False,indent=2),encoding='utf-8')
    return data, (xt,xv,xe,yt,yv,ye), summary, matrices, reports, oov, history, predictions

def figures(bundle):
    data,splits,summary,matrices,reports,oov,history,predictions = bundle
    fig,axs = plt.subplots(1,2,figsize=(7.2,2.45),layout='constrained')
    counts = data.label.value_counts().reindex(LABELS)
    bars = axs[0].bar(LABELS,counts,color=[COLORS[4],COLORS[2],COLORS[0]])
    axs[0].bar_label(bars,labels=[f'{n:,}\n({100*n/len(data):.1f}%)' for n in counts],padding=4,fontsize=8)
    axs[0].set_ylim(0,10500);axs[0].set_ylabel('Documents');axs[0].set_title('(a) NYT class imbalance')
    for j,(name,y) in enumerate(zip(['Train (9,215)','Val (1,152)','Test (1,152)'],splits[3:])):
        bottom=0
        for k,label in enumerate(LABELS):
            n=int((y==label).sum());pct=100*n/len(y)
            axs[1].barh(j,pct,left=bottom,color=[COLORS[4],COLORS[2],COLORS[0]][k],label=label if j==0 else None)
            axs[1].text(bottom+pct/2,j,str(n),ha='center',va='center',fontsize=8,color='white' if k else '#202020');bottom+=pct
    axs[1].set_yticks(range(3),['Train (9,215)','Val (1,152)','Test (1,152)']);axs[1].set_xlim(0,100);axs[1].set_xlabel('Class proportion (%)');axs[1].set_title('(b) Stratified split');axs[1].legend(ncol=3,loc='upper center',bbox_to_anchor=(.45,-.24),frameon=False)
    save(fig,'class_distribution')
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.65),layout='constrained')
    for ax,metric,title in zip(axs,['Test Accuracy','Test Macro-F1'],['(a) Accuracy','(b) Macro-F1']):
        vals=summary[metric].to_numpy()*100
        bars=ax.barh(np.arange(6),vals,color=COLORS,height=.65)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
        ax.set_yticks(range(6),METHODS);ax.invert_yaxis();ax.set_xlim(94 if 'Macro' in metric else 97,100.3);ax.set_xlabel('Score (%)');ax.set_title(title);ax.grid(axis='x',alpha=.2)
    save(fig,'performance')
    fig,axs=plt.subplots(2,3,figsize=(7.2,4.7),layout='constrained')
    for i,(ax,cm) in enumerate(zip(axs.flat,matrices)):
        norm=cm/cm.sum(axis=1,keepdims=True)
        heatmap(ax,norm,'Blues',1)
        for r in range(3):
            for c in range(3):
                ax.text(c,r,f'{cm[r,c]}\n{norm[r,c]*100:.1f}%',ha='center',va='center',fontsize=8,color='white' if norm[r,c]>.5 else '#263444')
        ax.set_xticks(range(3),['Bus.','Pol.','Spo.']);ax.set_yticks(range(3),['Bus.','Pol.','Spo.']);ax.set_title(f'{METHODS[i]} | {sum(cm.sum(axis=1)-cm.diagonal())} errors')
        ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.spines[['top','right','left','bottom']].set_visible(False)
    save(fig,'confusion_matrices')
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.25),layout='constrained')
    for ax,key,title in zip(axs,['Token OOV Rate','Unique OOV Rate'],['(a) Token OOV','(b) Type OOV']):
        bars=ax.bar(['NYT W2V','AG W2V','GloVe'],oov[key]*100,color=[COLORS[2],COLORS[3],COLORS[4]],width=.55)
        ax.bar_label(bars,fmt='%.2f%%',padding=3);ax.set_ylim(0,max(oov[key]*100)*1.22);ax.set_ylabel('OOV (%)');ax.set_title(title)
    save(fig,'oov')
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.35),layout='constrained')
    for key,color in [('Train Loss',COLORS[2]),('Validation Loss',COLORS[5])]:
        axs[0].plot(history.Epoch,history[key],'o-',color=color,label=key)
    axs[0].set_title('(a) Optimization and generalization');axs[0].set_ylabel('Cross-entropy loss');axs[0].legend(frameon=False)
    for key,color in [('Validation Accuracy',COLORS[2]),('Validation Macro-F1',COLORS[5])]:
        axs[1].plot(history.Epoch,history[key]*100,'o-',color=color,label=key)
    axs[1].axvline(2,color='#999999',ls='--',lw=.8);axs[1].set_title('(b) Best checkpoint: epoch 2');axs[1].set_ylabel('Validation score (%)');axs[1].legend(frameon=False,loc='center right')
    for ax in axs:ax.set_xticks([1,2,3]);ax.set_xlabel('Epoch');ax.grid(alpha=.15)
    save(fig,'bert_training')
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.8),layout='constrained')
    directions=[(0,1),(1,0),(0,2),(1,2),(2,0),(2,1)]
    error_counts=np.array([[cm[r,c] for r,c in directions] for cm in matrices])
    heatmap(axs[0],error_counts,'Oranges',error_counts.max(),aspect='auto')
    for r in range(6):
        for c in range(6):axs[0].text(c,r,str(error_counts[r,c]),ha='center',va='center',fontsize=8)
    axs[0].set_xticks(range(6),['B→P','P→B','B→S','P→S','S→B','S→P']);axs[0].set_yticks(range(6),METHODS);axs[0].set_title('(a) Error directions (counts)')
    vals=np.array([[cr[label]['f1-score']*100 for label in LABELS] for cr in reports])
    for j,label in enumerate(LABELS):axs[1].plot(range(6),vals[:,j],'o-',label=label,color=[COLORS[4],COLORS[2],COLORS[0]][j],ms=4)
    axs[1].set_xticks(range(6),['Bin.','Freq.','NYT','AG','GloVe','BERT'],rotation=25);axs[1].set_ylabel('Class F1 (%)');axs[1].set_title('(b) Minority-class performance');axs[1].legend(frameon=False);axs[1].grid(alpha=.15)
    save(fig,'error_analysis')
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.4),layout='constrained')
    words=pd.read_csv(ROOT/'task1_results/binary_bow_top_words.csv')
    for j,(ax,label) in enumerate(zip(axs,LABELS)):
        g=words[words['class']==label].head(6).iloc[::-1]
        ax.barh(g.word,g.coefficient,color=[COLORS[4],COLORS[2],COLORS[0]][j],height=.65);ax.set_title(label);ax.set_xlabel('LR coefficient');ax.grid(axis='x',alpha=.15)
    save(fig,'keywords')
    # Model agreement: reconstructed complete predictions, not only sampled errors.
    fig,ax=plt.subplots(figsize=(3.45,2.65),layout='constrained')
    joint=np.zeros((6,6),dtype=int);truth=np.array(splits[-1])
    for i in range(6):
        for j in range(6):joint[i,j]=np.sum((np.array(predictions[i])!=truth)&(np.array(predictions[j])!=truth))
    heatmap(ax,joint,'Blues',23)
    for i in range(6):
        for j in range(6):ax.text(j,i,str(joint[i,j]),ha='center',va='center',fontsize=8,color='white' if joint[i,j]>12 else '#263444')
    short=['Bin.','Freq.','NYT','AG','GloVe','BERT'];ax.set_xticks(range(6),short,rotation=30);ax.set_yticks(range(6),short);ax.set_title('Shared errors on the same test documents')
    save(fig,'error_overlap')

if __name__=='__main__':
    results=audit()
    figures(results)
    print(f'Passed {len(AUDIT["checks"])} checks; generated 8 vector PDF figures.')
    print(json.dumps({k:v for k,v in AUDIT.items() if k in ['data','metrics','binary_vs_frequency_disagreement','nyt_vs_ag_word2vec_disagreement','word_lengths','bert_epoch_seconds']},ensure_ascii=False,indent=2))
