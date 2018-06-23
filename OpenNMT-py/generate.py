#!/usr/bin/env python

from __future__ import division, unicode_literals
import os
import argparse
import math
import codecs
import torch
import numpy as np
from copy import deepcopy

from itertools import count
from tqdm import tqdm
import time

import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules
import opts
from pdb import set_trace

from drqa.reader import Predictor
from drqa import tokenizers

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts.add_md_help_argument(parser)
opts.generate_opts(parser)
opts.DrQA_opts(parser)

opt = parser.parse_args()

tokenizers.set_default('corenlp_classpath', opt.corenlp_path)

eps = 1e-15
from difflib import SequenceMatcher
from math import log

prob_weight = 1e-10
qa_weight = 1 - prob_weight
QA_scheme = 2

# opt.model = '/home/jack/Documents/QA_QG/lan-model/results/2.4.2018/model_acc_46.69_ppl_23.55_e12.pt'
# opt.batch_size = batch_size
opt.gpu = 0

def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    # Load the model.
    fields, model, model_opt = \
        onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__)

    # load DrQA model
    if opt.selection_criterion == 'NLL-QA':
        DrQA_predictor = Predictor(
            model=opt.DrQA_model,
            tokenizer='corenlp',
            # embedding_file=opt.embedding_file,
            # num_workers=opt.num_workers,
        )
        DrQA_predictor.cuda()

    # Test data
    data = onmt.io.build_dataset(fields, opt.data_type,
                                 opt.src, opt.tgt,
                                 src_dir=opt.src_dir,
                                 sample_rate=opt.sample_rate,
                                 window_size=opt.window_size,
                                 window_stride=opt.window_stride,
                                 window=opt.window,
                                 use_filter_pred=False)

    test_data = onmt.io.OrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=opt.batch_size, train=False, sort=False,
        shuffle=False)

    # Translator
    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha, opt.beta)
    translator = onmt.translate.Translator(model, fields,
                                           beam_size=opt.beam_size,
                                           n_best=opt.n_best,
                                           global_scorer=scorer,
                                           max_length=opt.max_sent_length,
                                           copy_attn=model_opt.copy_attn,
                                           cuda=opt.cuda,
                                           beam_trace=opt.dump_beam != "")
    builder = onmt.translate.TranslationBuilder(
        data, translator.fields,
        opt.n_best, opt.replace_unk, opt.tgt)

    # Statistics
    counter = count(1)
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0

    questions = []
    raw_prob_scores = []
    qa_scores = []
    QA_augmented_scores = []

    best_question_raw_prob = []
    best_question_QA_augmented = []
    best_raw_prob_scores = []
    best_QA_augmented_scores = []
    QA_choice_indices = []

    num_diff = 0

    counter = -1
    for batch in tqdm(test_data):
        counter += 1
        batch_data = translator.translate_batch(batch, data)
        translations = builder.from_batch(batch_data)

        # set_trace()

        ##########################
        # visualization block
        ##########################

        # get the answer word indices
        ansIdx = [i for i in range(len(data.examples[counter].src_feat_3))
                  if 'A' in data.examples[counter].src_feat_3[i]]
        selected_pred = translations[0].pred_sents[0]  # hard code select the first sentence, assuming it has lowest ppl
        selected_pred = [token.replace('\n', '') for token in selected_pred]
        if 'eos' not in selected_pred:
            selected_pred.append('eos')

        # visualization of saliency
        if opt.saliency_vis:
            # make animation that shows the change of saliency through output time steps
            # set_trace()
            all_saliency = batch_data['saliency']
            # sanity check
            # if torch.max(all_saliency) > 1 or torch.max(all_saliency) < -1:
            #     print(torch.max(all_saliency))
            #     print(torch.min(all_saliency))
            #     raise Exception('gradient value should be in range [-1, 1] (is this True???)')
            fig, ax = plt.subplots(figsize=(12, 10))
            # fig = plt.figure()
            im = ax.imshow(all_saliency[:,0].data.cpu().numpy(), aspect='auto', cmap='Spectral', animated=True)
            # ax = plt.axes()
            # title = ax.text(0, 0, selected_pred[0])
            plt.yticks(np.arange(len(batch.dataset.examples[counter].src)), batch.dataset.examples[counter].src)
            plt.colorbar(im)
            # set color/font of yticks
            for idx in ansIdx:
                plt.gca().get_yticklabels()[idx].set_color('blue')
            def init():
                im.set_array(np.ma.array(all_saliency[:,0], mask=True))
                return im,
            def update(i):
                im.set_array(all_saliency[:,i].data.cpu().numpy())
                # im.set_clim(vmin=torch.min(all_saliency[:,i]), vmax=torch.max(all_saliency[:,i]))
                ax.set_title(selected_pred[i])
                # fig.colorbar(im)
                # print(i)
                return im,
            ani = animation.FuncAnimation(fig, update, frames=range(len(selected_pred)), interval=1000)
            # set_trace()
            # plt.draw()
            # plt.show()
            ani.save("saliency_animations_LSTM_attn/saliency_sent_" +
                     str(counter+1) + "(" + opt.src.split('/')[1] + ").mp4")

        # visualization of attention matrix
        # set_trace()
        if opt.attn_vis:
            attn = batch_data['attention'][0][0].t()
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(attn.data.cpu().numpy(), aspect='auto', cmap='Greys')
            im.set_clim(vmin=torch.min(attn), vmax=torch.max(attn))
            plt.yticks(np.arange(len(batch.dataset.examples[counter].src)), batch.dataset.examples[counter].src)
            for idx in ansIdx:
                plt.gca().get_yticklabels()[idx].set_color('blue')
            plt.xticks(np.arange(len(selected_pred)), tuple(selected_pred), rotation=45)
            plt.colorbar(im)
            # set_trace()
            # plt.show()
            plt.savefig('attention_visualizations_LSTM_attn/attn_vis_' + str(counter+1) +
                        "(" + opt.src.split('/')[1] + ").png")

        ##########################
        # end of visualization block
        ##########################

        for trans in translations:
            pred_score_total += trans.pred_scores[0]
            pred_words_total += len(trans.pred_sents[0])
            if opt.tgt:
                gold_score_total += trans.gold_score
                gold_words_total += len(trans.gold_sent)


            ''' ranking with raw probability '''
            n_best_preds_raw_prob = [" ".join(pred).replace('\n', '')
                            for pred in trans.pred_sents[:opt.n_best] if '\n' not in pred]
            # n_best_raw_prob = [-trans.pred_scores[i] for i in range(opt.n_best)]
            n_best_raw_prob = [np.exp(trans.pred_scores[i]) for i in range(opt.n_best)] # raw prob, higher is better
            assert(len(n_best_preds_raw_prob) == opt.n_best)
            assert(len(n_best_raw_prob) == opt.n_best)

            # append best to list
            best_question_raw_prob.append(n_best_preds_raw_prob[0])


            ''' ranking with raw prob + QA '''
            if opt.selection_criterion == 'NLL-QA':
                # prepare DrQA inputs
                candidates = deepcopy(n_best_preds_raw_prob)

                # deal with situations where the candidate question is '', which QA does not process
                for i in range(len(candidates)):
                    if candidates[i] == '':
                        candidates[i] = '?'

                contexts = [' '.join(batch.dataset.examples[batch.indices[0].data[0]].src).strip()] * len(candidates)
                qc = [(contexts[k], candidates[k]) for k in range(len(candidates))]
                # recover answer words
                ans_idx = np.where(batch.src_feat_3.data.t().cpu().numpy()[0] == 3)[0]
                ans = ' '.join([batch.dataset.examples[batch.indices[0].data[0]].src[k] for k in ans_idx]).strip()

                # set_trace()
                # QA scoring
                qa_score = []
                # try:
                QA_results, examples = DrQA_predictor.predict_batch(qc, top_n=5) # NOTE top_n hard coded
                # except:
                #     set_trace()
                # top_n_raw_prob = [trans.pred_scores[i] for i in range(opt.n_best)]
                if QA_scheme == 1:
                    # scoring scheme 1: if one of the generated answers exactly matches the true answer, then use that
                    #                   confidence score for QA score calculation. If none of the generated answer
                    #                   matches the true answer exactly, then find the highest string match score,
                    #                   and multiply it with the confidence of that match
                    for result in QA_results: # for each question-context pair in the batch
                        HAS_ANS = False
                        for k in range(len(result)):
                            if result[k][0] == ans:
                                HAS_ANS = True
                                score = result[k][1]
                        if HAS_ANS:
                            qa_score.append(score)
                        else:
                            sim = [SequenceMatcher(None, ans, result[k][0]).ratio() for k in range(len(result))]
                            weighted_prob = max(sim) * [cand[1] for cand in result][sim.index(max(sim))] + eps
                            qa_score.append(-10 * log(weighted_prob))
                            # qa_score.append(weighted_prob)
                elif QA_scheme == 2:
                    # scoring scheme 2: multiply the generated answer confidence with the string match score
                    for result in QA_results:  # for each question-context pair in the batch
                        sim = [SequenceMatcher(None, ans, result[k][0]).ratio() for k in range(len(result))]
                        weighted_prob = np.array([sim[k] * result[k][1] + eps for k in range(len(result))])
                        qa_score.append(np.min(-10 * np.log(weighted_prob)))
                        # qa_score.append(np.max(weighted_prob)) # probability, higher is better
                        # set_trace()
                # set_trace()
                QA_augmented_score = qa_weight*np.array(qa_score) + prob_weight*(-np.log(n_best_raw_prob)) 

                # record all scores, questions (#: n_best * #src)
                raw_prob_scores += n_best_raw_prob
                qa_scores += qa_score
                QA_augmented_scores += QA_augmented_score.tolist()
                questions += candidates

                # record best scores, best questions (# = #src)
                best_raw_prob_scores.append(-n_best_raw_prob[0])
                best_QA_augmented_scores.append(min(QA_augmented_score))
                # best_QA_augmented_scores.append(max(QA_augmented_score))
                QA_choice_idx = np.where(QA_augmented_score == min(QA_augmented_score))[0][0]
                # QA_choice_idx = np.where(QA_augmented_score == max(QA_augmented_score))[0][0]
                best_question_QA_augmented.append(candidates[QA_choice_idx])
                QA_choice_indices.append(QA_choice_idx)

                # set_trace()

                # see if the qa_augmented score selection is different from raw prob selection
                if n_best_preds_raw_prob[0] != candidates[QA_choice_idx]:
                    num_diff += 1


                print('different best question produced using additional QA: ' + str(num_diff))

                # sanity check (lengths)
                # set_trace()
                assert(len(raw_prob_scores) == len(qa_scores) == len(QA_augmented_scores)
                       == len(questions))
                        # == opt.n_best*len(test_data.dataset.examples))
                assert(len(best_raw_prob_scores) == len(best_QA_augmented_scores) == len(best_question_raw_prob)
                       == len(best_question_QA_augmented))
                        # == len(test_data.dataset.examples))


    # write file for debugging
    # write all questions (n_best * )
    if opt.debug:
        with codecs.open(opt.output+'.DEBUG.'+str(prob_weight)+'.'+str(qa_weight)+'.scheme'+str(QA_scheme)+'.txt', 'w', 'utf-8') as debug_file:
            # for idx in range(len(raw_prob_scores)):
            #     debug_file.write(questions[idx]
            #                      + ' || prob: ' + str(raw_prob_scores[idx])
            #                      + ' || qa" ' + qa_scores[idx]
            #                      + '\n')
            debug_file.write('\n'.join([questions[idx]
                                        + ' || prob: ' + str(raw_prob_scores[idx])
                                        + ' || qa: ' + str(qa_scores[idx])
                                        for idx in range(len(raw_prob_scores))] ))
        debug_file.close()

    # write best questions judged by raw probability
    # write all questions (n_best * )
    if opt.selection_criterion == 'NLL':
        with codecs.open(opt.output+'.prob.txt', 'w', 'utf-8') as f:
            f.write('\n'.join(best_question_raw_prob))
        f.close()

    # write best questions judged by QA score + raw probability
    # write all questions (n_best * )
    if opt.selection_criterion == 'NLL-QA':
        with codecs.open(opt.output+'.qa.prob.'
                         + str(prob_weight)
                         + '.' + str(qa_weight)
                         + '.scheme' + str(QA_scheme)
                         + '.txt', 'w', 'utf-8') as f:
            f.write('\n'.join(best_question_QA_augmented))
        f.close()

    # write the choice to file
    if opt.debug:
        with codecs.open(opt.output+'.qa.prob.'
                         + str(prob_weight)
                         + '.'+str(qa_weight)
                         + '.scheme'
                         + str(QA_scheme)
                         + 'choice.indices.txt', 'w', 'utf-8') as f:
            f.write('\n'.join([str(n) for n in QA_choice_indices]))
        f.close()


if __name__ == "__main__":
    main()
