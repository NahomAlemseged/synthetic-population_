import yaml
import numpy as np
import pandas as pd
import torch

from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors


# =====================================================
# DEVICE
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")


# =====================================================
# CONFIG
# =====================================================

CONFIG_PATH = Path(
    "/content/synthetic-population_/config/params.yaml"
)

with open(CONFIG_PATH) as f:
    params = yaml.safe_load(f)



class PrivacyEval:


    def __init__(self):

        self.train_path = params["evaluate"]["input"][2]
        self.test_path = params["evaluate"]["input"][1]

        self.synth_path = params["generate_drg"]["input"][3]

        self.target = "APR_DRG"

        self.sample_size = 100000



    # =====================================================
    # LOAD DATA
    # =====================================================

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


        print("Train :",train.shape)
        print("Test  :",test.shape)
        print("Synth :",synth.shape)


        return train,test,synth



    # =====================================================
    # COMMON CLASSES
    # =====================================================

    def align_classes(
        self,
        train,
        test,
        synth
    ):


        common = (
            set(train[self.target])
            &
            set(test[self.target])
            &
            set(synth[self.target])
        )


        print(
            f"Keeping {len(common)} common APR_DRG classes"
        )


        train=train[
            train[self.target].isin(common)
        ]

        test=test[
            test[self.target].isin(common)
        ]

        synth=synth[
            synth[self.target].isin(common)
        ]


        return train,test,synth



    # =====================================================
    # ENCODING
    # =====================================================

    def encode(
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
            ignore_index=True
        )


        y=combined[self.target]

        X=combined.drop(
            columns=[self.target]
        )


        for col in X.select_dtypes(
            include="object"
        ).columns:


            le=LabelEncoder()


            X[col]=le.fit_transform(
                X[col].astype(str)
            )



        X=X.fillna(-1)



        n1=len(train)
        n2=len(test)



        X_train=X.iloc[:n1]
        X_test=X.iloc[n1:n1+n2]
        X_synth=X.iloc[n1+n2:]


        y_train=y.iloc[:n1]
        y_test=y.iloc[n1:n1+n2]
        y_synth=y.iloc[n1+n2:]


        return (
            X_train,
            X_test,
            X_synth,
            y_train,
            y_test,
            y_synth
        )



    # =====================================================
    # MIA
    # =====================================================

    def mia(
        self,
        X_train,
        X_test
    ):


        # sample for speed

        n=min(
            self.sample_size,
            len(X_train),
            len(X_test)
        )


        X_in=X_train.sample(
            n,
            random_state=42
        )


        X_out=X_test.sample(
            n,
            random_state=42
        )



        X=np.vstack(
            [
                X_in,
                X_out
            ]
        )


        y=np.concatenate(
            [
                np.ones(n),
                np.zeros(n)
            ]
        )


        X1,X2,y1,y2=train_test_split(
            X,
            y,
            test_size=.3,
            random_state=42
        )



        model=RandomForestClassifier(
            n_estimators=50,
            n_jobs=-1,
            random_state=42
        )


        model.fit(
            X1,
            y1
        )


        prob=model.predict_proba(
            X2
        )[:,1]


        auc=roc_auc_score(
            y2,
            prob
        )


        return {
            "mia_auc":auc
        }




    # =====================================================
    # NND
    # =====================================================

    def nnd(
        self,
        X_train,
        X_synth,
        X_test
    ):


        scaler=StandardScaler()


        X_train=scaler.fit_transform(
            X_train
        )


        X_synth=scaler.transform(
            X_synth
        )


        X_test=scaler.transform(
            X_test
        )



        # real-real baseline

        nn=NearestNeighbors(
            n_neighbors=1,
            n_jobs=-1
        )


        nn.fit(X_train)


        real_dist,_=nn.kneighbors(
            X_test
        )


        synth_dist,_=nn.kneighbors(
            X_synth
        )


        real_mean=np.mean(real_dist)

        synth_mean=np.mean(synth_dist)



        # privacy relative to real variation

        P_nnd=(
            synth_mean /
            (
                synth_mean+
                real_mean
            )
        )


        leak_rate=np.mean(
            synth_dist <
            np.percentile(
                real_dist,
                5
            )
        )


        return {

            "real_nnd":
            real_mean,


            "mean_nnd":
            synth_mean,


            "leak_rate":
            leak_rate,


            "P_nnd":
            P_nnd
        }




    # =====================================================
    # PRIVACY SCORE
    # =====================================================

    def privacy_score(
        self,
        result
    ):


        P_mia = (
            1-
            abs(
                2*
                (
                    result["mia_auc"]-0.5
                )
            )
        )


        P_leak = (
            1-result["leak_rate"]
        )


        score=(
            P_mia+
            result["P_nnd"]+
            P_leak
        )/3



        return {

            "P_mia":
            P_mia,

            "P_leak":
            P_leak,

            "privacy_score":
            score

        }



    # =====================================================
    # RUN
    # =====================================================

    def run(self):


        train,test,synth=self.load_data()


        train,test,synth=self.align_classes(
            train,
            test,
            synth
        )


        (
            X_train,
            X_test,
            X_synth,
            y_train,
            y_test,
            y_synth

        )=self.encode(
            train,
            test,
            synth
        )



        result={}



        print("\nRunning MIA")

        result.update(
            self.mia(
                X_train,
                X_test
            )
        )



        print("\nRunning NND")

        result.update(
            self.nnd(
                X_train,
                X_synth,
                X_test
            )
        )



        result.update(
            self.privacy_score(
                result
            )
        )



        print("\n========================")
        print("PRIVACY REPORT")
        print("========================")


        for k,v in result.items():

            print(
                f"{k:20}: {v:.4f}"
            )



        pd.DataFrame(
            [result]
        ).to_csv(
            "privacy_report.csv",
            index=False
        )


        return result




def main():

    evaluator=PrivacyEval()

    evaluator.run()



if __name__=="__main__":

    main()
