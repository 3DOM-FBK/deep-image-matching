from .LightGlue.lightglue import ALIKED

def aliked(num_keypoints=4096):
    model = ALIKED(num_keypoints=num_keypoints)
    model = model.eval()
    return model

detector = aliked()

def extract_aliked_kpts(img, device='cpu'):
    detector.to(device)

    pred = detector.extract(img)
    keypoints = pred['keypoints']
    scores = pred['keypoint_scores']
    
    return keypoints, scores