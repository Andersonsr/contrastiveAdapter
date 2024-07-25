import clip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df = pd.read_excel('dataset/dataset_clip_filter.xlsx')
    x = []
    y = []
    for i, row in df.iterrows():
        tokens = clip.tokenize(row['gt_text'], context_length=512)
        x.append(len(row['gt_text']))
        y.append(np.count_nonzero(tokens.squeeze()))

    plt.scatter(x, y, marker='x')
    plt.axhline(77, label='CLIP limit', color='purple')
    plt.axhline(248, label='Long-CLIP limit', color='darkgreen')
    plt.xlabel('caption length')
    plt.ylabel('number of tokens')
    plt.title('GeoQA dataset tokens vs caption length')
    plt.legend()
    plt.show()

    y = np.array(y)
    print(len(np.where(y < 78)[0]))
    print(f'mean {np.mean(y)}`')
    print(f'min {np.min(y)}')
    print(f'max {np.max(y)}')
    print(f'median {np.median(y)}')
