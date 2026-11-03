import argparse, json, random
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
from .config import *
from .data import load_data, split_data, UserDataset
from .model import UserBERTClassifier

def evaluate(model, loader, device):
    model.eval(); ys=[]; preds=[]; probs=[]
    with torch.no_grad():
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()}
            logits=model(batch["input_ids"], batch["attention_mask"], batch["post_mask"])
            p=torch.softmax(logits, 1)[:,1]
            probs.extend(p.cpu().tolist()); preds.extend((p>=.5).long().cpu().tolist())
            ys.extend(batch["label"].cpu().tolist())
    return ys, preds, probs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="data/user_level_text_dataset_clean.csv")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args=ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df=load_data(args.data); train_df,val_df,test_df=split_data(df)
    tokenizer=BertTokenizer.from_pretrained(BASE_MODEL)
    loaders=[DataLoader(UserDataset(x,tokenizer), batch_size=args.batch_size,
                        shuffle=(i==0)) for i,x in enumerate((train_df,val_df,test_df))]
    train_loader,val_loader,test_loader=loaders
    model=UserBERTClassifier().to(device)
    optimizer=AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn=torch.nn.CrossEntropyLoss()
    MODEL_DIR.mkdir(parents=True,exist_ok=True); OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    best=-1
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch={k:v.to(device) for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits=model(batch["input_ids"],batch["attention_mask"],batch["post_mask"])
            loss=loss_fn(logits,batch["label"]); loss.backward(); optimizer.step()
            losses.append(loss.item())
        ys,preds,_=evaluate(model,val_loader,device); vf1=f1_score(ys,preds,zero_division=0)
        print(f"epoch={epoch+1} train_loss={np.mean(losses):.4f} val_f1={vf1:.4f}")
        if vf1>best:
            best=vf1; torch.save(model.state_dict(),MODEL_DIR/"model.pt")
            tokenizer.save_pretrained(MODEL_DIR)
    model.load_state_dict(torch.load(MODEL_DIR/"model.pt",map_location=device))
    ys,preds,probs=evaluate(model,test_loader,device)
    metrics={"test_accuracy":accuracy_score(ys,preds),"test_f1":f1_score(ys,preds,zero_division=0),
             "confusion_matrix":confusion_matrix(ys,preds).tolist()}
    (OUTPUT_DIR/"results.json").write_text(json.dumps(metrics,indent=2))
    print(classification_report(ys,preds,zero_division=0)); print(metrics)
if __name__=="__main__": main()
