from ehr_gan import load_and_encode, train_gan
import os


def main():
    csv = os.path.join(os.path.dirname(__file__), 'mimic-demo-dataset.csv')
    df, enc, X, inds = load_and_encode(csv)
    print('Loaded', df.shape)
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'simple_gan.pt')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    G, D = train_gan(X, enc, epochs=50, batch_size=32, save_path=model_path)


if __name__ == '__main__':
    main()
