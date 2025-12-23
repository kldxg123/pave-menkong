import json
import os
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

# =================配置区域=================
# 1. 刚才生成的预测文件路径
PRED_FILE = "/home/app-ahr/PAVE/data/video_instruction_tuning/prediction/pave_prediction_final.json"
# 2. 原始标注文件路径 (Ground Truth)
GT_FILE = "/home/app-ahr/PAVE/data/video_instruction_tuning/avsd/mock_test_set4DSTC10-AVSD_from_DSTC7_singref.json"
# =========================================

def convert_avsd_gt_to_coco_format(avsd_file):
    """
    将 AVSD 的标注格式转换为 COCO 评测格式
    """
    with open(avsd_file, 'r') as f:
        data = json.load(f)
    
    # AVSD json 通常包含 'dialogs' 列表
    dialogs = data.get('dialogs', [])
    
    annotations = []
    images = []
    
    for idx, item in enumerate(dialogs):
        vid = item['image_id']
        # 获取最后一轮的正确答案 (Ground Truth Answer)
        # 注意：AVSD 数据集中，最后一轮通常是我们想要预测的目标
        last_turn = item['dialog'][-1]
        gt_answer = last_turn['answer']
        
        images.append({"id": vid})
        annotations.append({
            "image_id": vid,
            "caption": gt_answer,
            "id": idx
        })
        
    return {"images": images, "annotations": annotations}

def convert_pred_to_coco_format(pred_file):
    """
    将生成的预测文件转换为 COCO 结果格式
    """
    with open(pred_file, 'r') as f:
        preds = json.load(f)
        
    coco_preds = []
    for item in preds:
        coco_preds.append({
            "image_id": item['image_id'],
            "caption": item['caption']
        })
    return coco_preds

def main():
    print(f"Loading Ground Truth from: {GT_FILE}")
    print(f"Loading Predictions from: {PRED_FILE}")

    if not os.path.exists(PRED_FILE):
        print(f"[Error] 找不到预测文件 {PRED_FILE}。请先运行推理脚本生成结果！")
        return

    # 1. 转换数据格式
    gt_data = convert_avsd_gt_to_coco_format(GT_FILE)
    pred_data = convert_pred_to_coco_format(PRED_FILE)

    # 临时保存转换后的 GT 文件，供 COCO 类读取
    temp_gt_file = "temp_gt_coco_format.json"
    with open(temp_gt_file, "w") as f:
        json.dump(gt_data, f)

    # 2. 初始化 COCO 对象
    coco = COCO(temp_gt_file)
    coco_result = coco.loadRes(pred_data)

    # 3. 运行评测
    print("\nStarting Evaluation...")
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()
    coco_eval.evaluate()

    # 4. 打印最终结果
    print("\n" + "="*40)
    print("       FINAL ACCURACY METRICS       ")
    print("="*40)
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: \t{score:.4f}")
    print("="*40)

    # 清理临时文件
    if os.path.exists(temp_gt_file):
        os.remove(temp_gt_file)

if __name__ == "__main__":
    main()