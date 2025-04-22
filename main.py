from utils.data_preprocess import load_and_preprocess
from utils.dataset import create_dataloaders
from models.lstm_model import TrafficVolumeLSTM
from train import train_model
from evaluate import evaluate

def main():
    X, y, scaler = load_and_preprocess("data/Metro_Interstate_Traffic_Volume.csv")
    train_loader, val_loader, test_loader = create_dataloaders(X, y)

    model = TrafficVolumeLSTM()
    train_model(model, train_loader, val_loader)
    evaluate(model, test_loader, scaler)

if __name__ == "__main__":
    main()