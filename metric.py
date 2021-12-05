import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from skimage.metrics import structural_similarity as ssim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calc_anomaly_score_one_frame(frame_true, frame_hat, flow_true, flow_hat, thresh_cut_off=[0, 0, 0], return_as_map=False, operation=np.mean):
    assert frame_true.shape == frame_hat.shape
    assert flow_true.shape == flow_hat.shape

    loss_appe = (frame_true-frame_hat)**2
    loss_flow = (flow_true-flow_hat)**2

    # calc angle of optical flow
    _, angle_true = cv2.cartToPolar(flow_true[:, :, 0], flow_true[:, :, 1])
    _, angle_hat = cv2.cartToPolar(flow_hat[:, :, 0], flow_hat[:, :, 1])
    angle_diff = np.absolute(angle_true - angle_hat)
    loss_angle_flow = np.min(np.array([angle_diff, 2*np.pi - angle_diff]), axis=0)**2
    assert loss_angle_flow.shape == flow_true.shape[:2]

    # cut-off low scores to check only high scores
    if thresh_cut_off is not None:
        assert len(thresh_cut_off) == 3
        loss_appe = np.clip(loss_appe, thresh_cut_off[0], None)
        loss_flow = np.clip(loss_flow, thresh_cut_off[1], None)
        loss_angle_flow = np.clip(loss_angle_flow, thresh_cut_off[2], None)

    # return score map for pixel-wise assessment
    if return_as_map:
        return operation(loss_appe, axis=-1), operation(loss_flow, axis=-1), loss_angle_flow

    def calc_measures_single_item(item_true, item_hat, squared_error, max_val_hat):
        PSNR_X = 10*np.log10(np.max(item_hat)**2/np.mean(squared_error))
        PSNR_inv = np.max(item_hat)**2 * np.mean(squared_error)
        PSNR = 10*np.log10(max_val_hat**2/np.mean(squared_error))
        SSIM = ssim(item_true, item_hat, data_range=np.max([item_true, item_hat])-np.min([item_true, item_hat]),
                    multichannel=len(item_true.shape) == 3 and item_true.shape[-1] > 1)
        stat_MSE = np.mean(squared_error)
        stat_maxSE = np.max(squared_error)
        stat_std = np.std(squared_error)
        stat_MSE_1channel = np.mean(np.sum(squared_error, axis=-1)) if len(squared_error.shape) == 3 else -1
        stat_maxSE_1channel = np.max(np.sum(squared_error, axis=-1)) if len(squared_error.shape) == 3 else -1
        stat_std_1channel = np.std(np.sum(squared_error, axis=-1)) if len(squared_error.shape) == 3 else -1
        return np.array([PSNR_X, PSNR_inv, PSNR, SSIM, stat_MSE, stat_maxSE, stat_std, stat_MSE_1channel, stat_maxSE_1channel, stat_std_1channel])

    scores_appe = calc_measures_single_item(frame_true, frame_hat, loss_appe, 1.0)
    scores_flow = calc_measures_single_item(flow_true, flow_hat, loss_flow, np.pi)
    scores_angle = calc_measures_single_item(angle_true, angle_hat, loss_angle_flow, np.pi)
    scores_mag = calc_measures_single_item(flow_true[..., -1], flow_hat[..., -1], (flow_true[..., -1] - flow_hat[..., -1])**2, 20.)

    return np.array([scores_appe, scores_flow, scores_angle, scores_mag])

def basic_assess_AUC(scores, labels, plot_pr_idx=None):
    assert len(scores) == len(labels)
    if plot_pr_idx is not None:
        precision, recall, _ = precision_recall_curve(labels, scores[:, plot_pr_idx])
        print(len(np.where(labels == 0)[0]), len(np.where(labels == 1)[0]), len(np.unique(precision)), len(np.unique(recall)))
        step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        # try:
        #     plt.show()
        # except:
        plt.savefig('precision_recall.png')
    # print(scores.shape)
    # print( [ scores[:, i] for i in range(scores.shape[1])])
    auc = []
    pr = []
    for i in range(scores.shape[1]):
      try:
        auc.append(roc_auc_score(labels, scores[:, i]))
        pr.append(average_precision_score(labels, scores[:, i]))
      except:
        print(i)
        pass
    return auc , pr

def full_assess_AUC(score_frame, frame_labels, w_img=0.5, w_flow=0.5, sequence_n_frame=None,
                    clip_normalize=True, use_pr=False, selected_score_estimation_ways=None, save_pr_appe_SSIM_epoch=None):
    def normalize_clip_scores(scores, ver=1):
        assert ver in [1, 2]
        if ver == 1:
            return [item/np.max(item, axis=0) for item in scores]
        else:
            return [(item-np.min(item, axis=0))/(np.max(item, axis=0)-np.min(item, axis=0)) for item in scores]

    scores_appe = score_frame[:, 0, :]
    scores_flow = score_frame[:, 1, :]
    scores_angle = score_frame[:, 2, :]
    scores_mag = score_frame[:, 3, :]

    need_append = len(score_frame) < len(frame_labels)

    idx = selected_score_estimation_ways
    if idx is None:
        idx = np.arange(scores_appe.shape[1])
    scores_appe = scores_appe[:, idx]
    scores_flow = scores_flow[:, idx]
    scores_angle = scores_angle[:, idx]
    scores_mag = scores_mag[:, idx]

    if not isinstance(w_img, float):
        w_img = w_img[idx]
    if not isinstance(w_flow, float):
        w_flow = w_flow[idx]
    # w_img, w_flow = abs(w_img), abs(w_flow)
    print('shape:', w_img.shape, w_flow.shape, scores_appe.shape, scores_flow.shape)
    scores_comb = np.log(1./w_img**1*scores_appe) + 2*np.log(1./w_flow**1*scores_flow)
    # scores_comb *= -1
    if sequence_n_frame is not None:
        accumulated_n_frame = np.cumsum(sequence_n_frame-1)[:-1]

        scores_appe = np.split(scores_appe, accumulated_n_frame, axis=0)
        scores_flow = np.split(scores_flow, accumulated_n_frame, axis=0)
        scores_comb = np.split(scores_comb, accumulated_n_frame, axis=0)
        scores_angle = np.split(scores_angle, accumulated_n_frame, axis=0)
        scores_mag = np.split(scores_mag, accumulated_n_frame, axis=0)

        if clip_normalize:
            ver = 1
            np.seterr(divide='ignore', invalid='ignore')
            scores_appe = normalize_clip_scores(scores_appe, ver=ver)
            scores_flow = normalize_clip_scores(scores_flow, ver=ver)
            scores_comb = normalize_clip_scores(scores_comb, ver=ver)
            scores_angle = normalize_clip_scores(scores_angle, ver=ver)
            scores_mag = normalize_clip_scores(scores_mag, ver=ver)

        if need_append:
            scores_appe = [np.concatenate((item, [item[0]]), axis=0) for item in scores_appe]
            scores_flow = [np.concatenate((item, [item[0]]), axis=0) for item in scores_flow]
            scores_comb = [np.concatenate((item, [item[0]]), axis=0) for item in scores_comb]
            scores_angle = [np.concatenate((item, [item[0]]), axis=0) for item in scores_angle]
            scores_mag = [np.concatenate((item, [item[0]]), axis=0) for item in scores_mag]

        scores_appe = np.concatenate(scores_appe, axis=0)
        scores_flow = np.concatenate(scores_flow, axis=0)
        scores_comb = np.concatenate(scores_comb, axis=0)
        scores_angle = np.concatenate(scores_angle, axis=0)
        scores_mag = np.concatenate(scores_mag, axis=0)

    print(scores_appe.shape, scores_flow.shape, scores_comb.shape, scores_angle.shape, scores_mag.shape)
    print('              PSNR_X,PSNR_inv,PSNR,SSIM,MSE,maxSE,std,MSE_1c,maxSE_1c,std_1c')
    print(scores_appe.shape)
    auc, prc = basic_assess_AUC(scores_appe, frame_labels) if len(np.unique(frame_labels)) > 1 else [-1, -1]
    print('appearance PRscore:', ', '.join(('%.3f' % val) for val in prc))
    print('appearance AUCs:', ', '.join(('%.3f' % val) for val in auc))
    appe_auc, appe_prc = auc, prc
    print("="*40)
    auc, prc = basic_assess_AUC(scores_flow, frame_labels) if len(np.unique(frame_labels)) > 1 else [-1, -1]
    print('optic flow PRscore:', ', '.join(('%.3f' % val) for val in prc))
    print('optic flow AUCs:', ', '.join(('%.3f' % val) for val in auc))
    print("="*40)
    auc, prc = basic_assess_AUC(scores_comb, frame_labels) if len(np.unique(frame_labels)) > 1 else [-1, -1]
    print('combinatio PRscore:', ', '.join(('%.3f' % val) for val in prc))
    print('combinatio AUCs:', ', '.join(('%.3f' % val) for val in auc))
    print("="*40)
    auc, prc = basic_assess_AUC(scores_angle, frame_labels) if len(np.unique(frame_labels)) > 1 else [-1, -1]
    print('direction  PRscore:', ', '.join(('%.3f' % val) for val in prc))
    print('direction  AUCs:', ', '.join(('%.3f' % val) for val in auc))
    print("="*40)
    auc, prc = basic_assess_AUC(scores_mag, frame_labels) if len(np.unique(frame_labels)) > 1 else [-1, -1]
    print('magnitude  PRscore:', ', '.join(('%.3f' % val) for val in prc))
    print('magnitude  AUCs:', ', '.join(('%.3f' % val) for val in auc))

    if save_pr_appe_SSIM_epoch is not None:
        p, r, _ = precision_recall_curve(frame_labels, scores_appe[:, 3])
        pr = [p, r]
        print('mAP of appearance SSIM:', average_precision_score(frame_labels, scores_appe[:, 3]))
    return scores_appe,scores_flow,scores_comb,scores_angle,scores_mag, appe_auc, appe_prc


def flip_scores(scores):
    norm_scores = np.zeros_like(scores)
    for i in range(len(norm_scores)):
        norm_scores[i] = scores[i]
        norm_scores[i, :, 0] = 1./norm_scores[i, :, 0]  # PSNR_X
        norm_scores[i, :, 2] = 1./norm_scores[i, :, 2]  # PSNR  
        #norm_scores[i, :, 2] = anomaly_score_inv(norm_scores[i, :, 2],norm_scores[i, :, 2].max(),norm_scores[i, :, 2].min())
        norm_scores[i, :, 3] = 1./norm_scores[i, :, 3]  # SSIM
        # norm_scores[i,:,6] *= norm_scores[i,:,4]**1
        # norm_scores[i,:,9] *= norm_scores[i,:,7]**1
    return norm_scores

def score_norm( score, quantile=0.95, output_min=0, output_max = 1 ,log=False, max_value=1):
  if quantile is not None:
    max_quantile_score = np.quantile(score.reshape(-1),quantile)
    print("max score: ", max_quantile_score)
    score = np.where(score >= max_quantile_score, max_quantile_score , score)
  if log:
    #score = np.clip(score,0,None)
    # print(score.min())
    # score = 1/(10*np.log10((max_value)**2/(score+1e-5)))
    # print(score.min(),score.max())
    score = np.log(score)
  
  plt.hist(score,bins=100)
#   try:
#       plt.show()
#   except:
  plt.savefig("Score_hist_"+str(quantile)+str(output_min)+str(output_max)+".png")
  max_score = np.max(score, axis=0)
  min_score = np.min(score, axis=0)
  range_score = max_score-min_score
  score = (score-min_score)/range_score
  score = score*(output_max-output_min)+output_min
  return score

def normalize_maxmin_scores(scores):
    max_ = np.max(scores, axis=0)
    min_ = np.min(scores, axis=0)
    return (scores-min_)/(max_-min_)

