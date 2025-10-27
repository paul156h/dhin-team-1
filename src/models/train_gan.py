import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.ehr_gan import load_and_encode, train_gan


def main():
    base = os.path.dirname(__file__)
    csv = os.path.join(base, '..', '..', 'data', 'datasets', 'mimic-demo-dataset.csv')
    df, enc, X, inds = load_and_encode(csv)
    print('Loaded', df.shape)
    model_path = os.path.join(os.path.dirname(__file__), 'simple_gan.pt')
    G, D = train_gan(X, enc, epochs=50, batch_size=32, save_path=model_path)


if __name__ == '__main__':
    main()
