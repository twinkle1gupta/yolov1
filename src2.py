import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def iou(box, clusters):
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    return intersection / (box_area + cluster_area - intersection + 1e-10)

def avg_iou(boxes, clusters):
    return np.mean([np.max(iou(box, clusters)) for box in boxes])

def kmeans(boxes, k, dist=np.median):
    rows = boxes.shape[0]
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))
    
    np.random.seed(42)
    clusters = boxes[np.random.choice(rows, k, replace=True)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)
        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

def run_anchor_box_pipeline(df, k=9):
    if all(col in df.columns for col in ['xmin', 'ymin', 'xmax', 'ymax']):
        df['width'] = df['xmax'] - df['xmin']
        df['height'] = df['ymax'] - df['ymin']
    boxes = df[['width', 'height']].values
    clusters = kmeans(boxes, k)
    avg = avg_iou(boxes, clusters)

    # Plotting (same style you used)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=boxes[:, 0], y=boxes[:, 1], label='Bounding Boxes')
    sns.scatterplot(x=clusters[:, 0], y=clusters[:, 1], color='red', marker='X', s=100, label='Anchor Boxes')
    plt.title(f"{k} Anchor Boxes")
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("anchor_boxes.png")

    return {
        "clusters": clusters.tolist(),
        "avg_iou": avg
    }
