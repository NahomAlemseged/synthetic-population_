import yaml
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors



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


        print("\nStarting Privacy Evaluation")


        self.train_path = params["evaluate"]["input"][2]
        self.test_path = params["evaluate"]["input"][1]
        self.synth_path = params["evaluate"]["input"][0]
        self.model_path = params["evaluate"]["input"][3]


        self.sample_size = 100000



    # =====================================================
    # LOAD DATA
    # =====================================================

    def load_data(self):


        print("\nLoading datasets")


        train=pd.read_csv(
            self.train_path,
            low_memory=False
        )


        test=pd.read_csv(
            self.test_path,
            low_memory=False
        )


        synth=pd.read_csv(
            self.synth_path,
            low_memory=False
        )


        print("Train :",train.shape)
        print("Test  :",test.shape)
        print("Synth :",synth.shape)


        return train,test,synth




    # =====================================================
    # LOAD MODEL
    # =====================================================

    def load_model(self):


        print("\nLoading model")


        bundle=joblib.load(
            self.model_path
        )


        print(
            "Bundle:",
            bundle.keys()
        )


        self.features=bundle["features"]


        print(
            "Model features:",
            self.features
        )




    # =====================================================
    # PREPARE FEATURES
    # SAME LOGIC FOR TRAIN/TEST/SYNTH
    # =====================================================

    def preprocess(
            self,
            df
    ):


        df=df.copy()


        # make missing features

        for c in self.features:

            if c not in df.columns:

                df[c]="UNKNOWN"



        X=df[
            self.features
        ].copy()



        # age conversion

        if "PAT_AGE" in X.columns:


            X["PAT_AGE"]=(
                X["PAT_AGE"]
                .astype(str)
                .str.extract(
                    r"(\d+)"
                )[0]
            )


            X["PAT_AGE"]=pd.to_numeric(
                X["PAT_AGE"],
                errors="coerce"
            )



        # encode object columns

        for col in X.columns:


            if X[col].dtype=="object":


                le=LabelEncoder()


                X[col]=le.fit_transform(
                    X[col]
                    .astype(str)
                )



        X=X.fillna(-1)



        return X





    # =====================================================
    # MIA
    # =====================================================

    def run_mia(
            self,
            X_train,
            X_test
    ):


        print("\nRunning MIA")


        n=min(
            self.sample_size,
            len(X_train),
            len(X_test)
        )


        member=X_train.sample(
            n,
            random_state=42
        )


        non_member=X_test.sample(
            n,
            random_state=42
        )



        X=np.vstack(
            [
                member,
                non_member
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
            test_size=0.3,
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


        prob=clf.predict_proba(
            X2
        )[:,1]


        auc=roc_auc_score(
            y2,
            prob
        )


        print(
            "MIA AUC:",
            auc
        )


        return auc





    # =====================================================
    # NND + LEAKAGE
    # =====================================================

    def run_nnd(
            self,
            X_train,
            X_synth,
            X_test
    ):


        print(
            "\nRunning nearest neighbor analysis"
        )


        scaler=StandardScaler()



        train=scaler.fit_transform(
            X_train
        )


        synth=scaler.transform(
            X_synth
        )


        test=scaler.transform(
            X_test
        )



        nn=NearestNeighbors(
            n_neighbors=1,
            n_jobs=-1
        )


        nn.fit(
            train
        )



        # synthetic nearest distance

        synth_dist,_=nn.kneighbors(
            synth
        )


        # real-real baseline

        real_dist,_=nn.kneighbors(
            test
        )



        synth_mean=np.mean(
            synth_dist
        )


        real_mean=np.mean(
            real_dist
        )



        threshold=np.percentile(
            real_dist,
            5
        )


        leak_rate=np.mean(
            synth_dist < threshold
        )



        print(
            "Synthetic NND:",
            synth_mean
        )


        print(
            "Leak rate:",
            leak_rate
        )


        return {

            "mean_nnd":
            synth_mean,

            "real_nnd":
            real_mean,

            "leak_rate":
            leak_rate
        }




    # =====================================================
    # PRIVACY SCORE
    # =====================================================

    def calculate_score(
            self,
            mia_auc,
            nnd_results
    ):


        # MIA

        P_MIA=max(
            0,
            1-
            2*
            abs(
                mia_auc-0.5
            )
        )



        # NND

        P_NND=(
            nnd_results["mean_nnd"]
            /
            (
                nnd_results["mean_nnd"]
                +
                1
            )
        )



        # leakage

        P_LEAKAGE=(
            1-
            nnd_results["leak_rate"]
        )



        score=np.mean(
            [
                P_MIA,
                P_NND,
                P_LEAKAGE
            ]
        )



        return {


            "P_MIA":
            P_MIA,


            "P_NND":
            P_NND,


            "P_LEAKAGE":
            P_LEAKAGE,


            "privacy_score":
            score*100

        }





    # =====================================================
    # RUN
    # =====================================================

    def run(self):


        train,test,synth=self.load_data()


        self.load_model()



        X_train=self.preprocess(
            train
        )


        X_test=self.preprocess(
            test
        )


        X_synth=self.preprocess(
            synth
        )



        print(
            "\nShapes:"
        )

        print(
            X_train.shape,
            X_test.shape,
            X_synth.shape
        )



        mia_auc=self.run_mia(
            X_train,
            X_test
        )



        nnd_results=self.run_nnd(
            X_train,
            X_synth,
            X_test
        )



        result=self.calculate_score(
            mia_auc,
            nnd_results
        )



        print(
            "\n======================"
        )

        print(
            "PRIVACY REPORT"
        )

        print(
            "======================"
        )


        for k,v in result.items():

            print(
                f"{k}: {v:.4f}"
            )



        pd.DataFrame(
            [
                {
                    **result,
                    **nnd_results,
                    "mia_auc":mia_auc
                }
            ]
        ).to_csv(
            "privacy_report.csv",
            index=False
        )



        print(
            "\nSaved privacy_report.csv"
        )




def main():

    evaluator=PrivacyEval()

    evaluator.run()



if __name__=="__main__":

    main()
