import yaml
import numpy as np
import pandas as pd
import mlflow
import torch

from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier


# =====================================================
# DEVICE
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"⚡ Using device: {DEVICE}")


# =====================================================
# CONFIG
# =====================================================

CONFIG_PATH = Path(
    "/content/synthetic-population_/config/params.yaml"
)

with open(CONFIG_PATH) as f:
    params = yaml.safe_load(f)



# =====================================================
# CLASS
# =====================================================

class PrivacyEval:


    def __init__(self):

        self.train_path = params["evaluate"]["input"][2]
        self.test_path = params["evaluate"]["input"][1]

        self.synth_path = params["generate_drg"]["input"][3]


        self.target = "APR_DRG"

        self.sample_size = 100000

        self.encoders = {}



    # =================================================
    # LOAD
    # =================================================

    def load_data(self):

        train = pd.read_csv(
            self.train_path,
            low_memory=False
        )

        test = pd.read_csv(
            self.test_path,
            low_memory=False
        )

        synth = pd.read_csv(
            self.synth_path,
            low_memory=False
        )


        for df in [train,test,synth]:

            if "PRINC_DIAG_CODE" in df.columns:

                df.drop(
                    columns=["PRINC_DIAG_CODE"],
                    inplace=True
                )


        print(
            f"Train : {train.shape}"
        )

        print(
            f"Test  : {test.shape}"
        )

        print(
            f"Synth : {synth.shape}"
        )


        return train,test,synth




    # =================================================
    # ALIGN DATASETS
    # =================================================

    def align_data(
            self,
            train,
            test,
            synth
    ):


        # common columns

        common_cols = (

            set(train.columns)
            &
            set(test.columns)
            &
            set(synth.columns)

        )


        common_cols.add(self.target)


        common_cols=list(common_cols)



        train=train[common_cols].copy()

        test=test[common_cols].copy()

        synth=synth[common_cols].copy()



        # keep common APR_DRG classes

        c1=set(train[self.target].unique())

        c2=set(test[self.target].unique())

        c3=set(synth[self.target].unique())


        common_y = c1 & c2 & c3



        print(
            f"Keeping {len(common_y)} common APR_DRG classes"
        )


        train=train[
            train[self.target].isin(common_y)
        ]

        test=test[
            test[self.target].isin(common_y)
        ]

        synth=synth[
            synth[self.target].isin(common_y)
        ]



        return train,test,synth




    # =================================================
    # ENCODE
    # =================================================

    def encode_all(
            self,
            train,
            test,
            synth
    ):


        combined=pd.concat(
            [
                train,
                test,
                synth
            ],
            axis=0
        )


        X_all=combined.drop(
            columns=[self.target]
        )


        y_all=combined[self.target]



        # categorical encoding

        for col in X_all.select_dtypes(
            include="object"
        ).columns:


            le=LabelEncoder()


            X_all[col]=le.fit_transform(
                X_all[col].astype(str)
            )


            self.encoders[col]=le




        X_all=X_all.fillna(-1)



        n1=len(train)

        n2=len(test)

        n3=len(synth)



        X_train=X_all.iloc[:n1]

        X_test=X_all.iloc[n1:n1+n2]

        X_synth=X_all.iloc[n1+n2:]



        y_train=y_all.iloc[:n1]

        y_test=y_all.iloc[n1:n1+n2]

        y_synth=y_all.iloc[n1+n2:]



        return (
            X_train,
            y_train,
            X_test,
            y_test,
            X_synth,
            y_synth
        )




    # =================================================
    # MIA
    # =================================================

    def mia(
            self,
            X_train,
            X_test
    ):


        X=np.vstack(
            [
                X_train,
                X_test
            ]
        )


        y=np.concatenate(
            [
                np.ones(len(X_train)),
                np.zeros(len(X_test))
            ]
        )


        X1,X2,y1,y2=train_test_split(
            X,
            y,
            test_size=.3,
            random_state=42
        )



        clf=RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            random_state=42
        )


        clf.fit(
            X1,
            y1
        )


        p=clf.predict_proba(
            X2
        )[:,1]


        return {

            "mia_auc":
            roc_auc_score(y2,p)

        }



    # =================================================
    # NND
    # =================================================

    def nnd(
            self,
            X_train,
            X_synth
    ):


        scaler=StandardScaler()


        X_train=scaler.fit_transform(
            X_train
        )


        X_synth=scaler.transform(
            X_synth
        )



        nn=NearestNeighbors(
            n_neighbors=1,
            n_jobs=-1
        )


        nn.fit(X_train)


        d,_=nn.kneighbors(
            X_synth
        )


        d=d.flatten()



        return {

            "mean_nnd":
            float(d.mean()),


            "median_nnd":
            float(np.median(d)),


            "leak_rate":
            float(
                np.mean(
                    d <
                    np.percentile(d,5)
                )
            )

        }



    # =================================================
    # UTILITY
    # =================================================

    def utility(
            self,
            X_synth,
            y_synth,
            X_test,
            y_test
    ):


        model=XGBClassifier(
            objective="multi:softmax",
            n_estimators=100,
            eval_metric="mlogloss",
            tree_method="hist",
            device=DEVICE,
            random_state=42
        )


        model.fit(
            X_synth,
            y_synth
        )


        pred=model.predict(
            X_test
        )


        return {

            "synth_real_accuracy":
            accuracy_score(
                y_test,
                pred
            ),


            "synth_real_f1":
            f1_score(
                y_test,
                pred,
                average="weighted"
            )

        }



    # =================================================
    # PRIVACY SCORE
    # =================================================

    def score(
            self,
            r
    ):


        P_mia = (
            1-
            abs(
                2*
                (
                    r["mia_auc"]-0.5
                )
            )
        )


        P_nnd = (
            r["mean_nnd"]
            /
            (
                r["mean_nnd"]+1
            )
        )


        P_leak = (
            1-r["leak_rate"]
        )



        return {

            "P_mia":P_mia,

            "P_nnd":P_nnd,

            "P_leak":P_leak,


            "privacy_score":
            (
                .4*P_mia+
                .3*P_nnd+
                .3*P_leak
            )

        }



    # =================================================
    # RUN
    # =================================================

    def run(self):


        train,test,synth=self.load_data()



        train,test,synth=self.align_data(
            train,
            test,
            synth
        )


        (
            X_train,
            y_train,
            X_test,
            y_test,
            X_synth,
            y_synth

        )=self.encode_all(
            train,
            test,
            synth
        )


        print("\n🔐 MIA")

        result={}

        result.update(
            self.mia(
                X_train.values,
                X_test.values
            )
        )



        print("\n📏 NND")

        result.update(
            self.nnd(
                X_train.values,
                X_synth.values
            )
        )



        # print("\n🧪 Utility")

        # result.update(
        #     self.utility(
        #         X_synth,
        #         y_synth,
        #         X_test,
        #         y_test
        #     )
        # )



        result.update(
            self.score(result)
        )



        print(
            "\n=========================="
        )

        print(
            "PRIVACY REPORT"
        )

        print(
            "=========================="
        )


        for k,v in result.items():

            print(
                f"{k:25}: {v:.4f}"
            )



        pd.DataFrame(
            [result]
        ).to_csv(
            "privacy_report.csv",
            index=False
        )


        return result




# =====================================================
# MAIN
# =====================================================

def main():

    evaluator=PrivacyEval()

    evaluator.run()



if __name__=="__main__":

    main()
