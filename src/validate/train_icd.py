import yaml
import joblib
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)

from xgboost import XGBClassifier



# ======================================================
# CONFIG
# ======================================================

CONFIG_PATH = Path(
    "/content/synthetic-population_/config/params.yaml"
)


with open(CONFIG_PATH) as f:
    params = yaml.safe_load(f)



INPUTS = params["evaluate"]["input"]

OUTPUT = Path(
    params["train"]["output"][1]
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_PATH = (
    OUTPUT /
    "tabddpm_tstr_icd.pkl"
)



# ======================================================
# ICD TRAINER
# ======================================================

class ICDTrainer:


    def __init__(self):

        self.synthetic_path = INPUTS[1]

        self.test_path = (
            "/content/drive/MyDrive/data_THCIC/gold/test.csv"
        )


        self.target = "PRINC_DIAG_CODE"



        # SAME FEATURES AFTER DRG GENERATION

        self.features = [

            "SEX_CODE",
            "PAT_AGE",
            "RACE",
            "ETHNICITY",
            "PAT_ZIP",
            "PAT_COUNTY",
            "PUBLIC_HEALTH_REGION",
            "APR_MDC",
            "APR_DRG"

        ]


        self.cat_features = [

            "SEX_CODE",
            "RACE",
            "ETHNICITY",
            "PAT_ZIP",
            "PAT_COUNTY",
            "PUBLIC_HEALTH_REGION",
            "APR_MDC",
            "APR_DRG"

        ]


        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )



    # ==================================================
    # LOAD
    # ==================================================

    def load(self,path):

        print(
            "Loading:",
            path
        )

        return pd.read_csv(
            path,
            low_memory=False
        )



    # ==================================================
    # CLEAN
    # ==================================================

    def clean(self,df):

        df=df.copy()


        df=df[
            df[self.target].notna()
        ]


        df[self.target]=(
            df[self.target]
            .astype(str)
            .str.strip()
        )


        df=df[
            df[self.target]!="UNKNOWN"
        ]



        # AGE

        df["PAT_AGE"]=(
            df["PAT_AGE"]
            .astype(str)
            .str.replace(
                r"\D",
                "",
                regex=True
            )
        )


        df["PAT_AGE"]=pd.to_numeric(
            df["PAT_AGE"],
            errors="coerce"
        )


        df=df.dropna(
            subset=self.features+[self.target]
        )


        return df




    # ==================================================
    # ENCODE
    # ==================================================

    def encode(self,train,test):


        self.encoder.fit(
            train[self.cat_features]
            .astype(str)
        )


        train[self.cat_features]=(
            self.encoder.transform(
                train[self.cat_features]
                .astype(str)
            )
        )


        test[self.cat_features]=(
            self.encoder.transform(
                test[self.cat_features]
                .astype(str)
            )
        )


        return train,test



    # ==================================================
    # TRAIN
    # ==================================================

    def run(self):


        print("\n==============================")
        print(" ICD TSTR TRAINING ")
        print("==============================")



        synthetic=self.load(
            self.synthetic_path
        )


        test=self.load(
            self.test_path
        )



        synthetic=self.clean(
            synthetic
        )


        test=self.clean(
            test
        )



        print(
            "Synthetic:",
            synthetic.shape
        )


        print(
            "Test:",
            test.shape
        )



        # keep common ICD classes

        common = (

            set(
                synthetic[self.target]
            )

            &

            set(
                test[self.target]
            )

        )


        print(
            "Common ICD classes:",
            len(common)
        )


        synthetic=synthetic[
            synthetic[self.target]
            .isin(common)
        ]


        test=test[
            test[self.target]
            .isin(common)
        ]



        # Encode

        synthetic,test=self.encode(
            synthetic,
            test
        )



        X_train=synthetic[
            self.features
        ]


        y_train=synthetic[
            self.target
        ]



        X_test=test[
            self.features
        ]


        y_test=test[
            self.target
        ]



        # convert ICD labels

        labels=sorted(
            y_train.unique()
        )


        label_map={
            v:i
            for i,v in enumerate(labels)
        }


        reverse_map={
            i:v
            for v,i in label_map.items()
        }



        y_train=y_train.map(
            label_map
        )


        y_test=y_test.map(
            label_map
        )


        valid=y_test.notna()


        X_test=X_test.loc[valid]

        y_test=y_test.loc[valid]



        print(
            "Training rows:",
            len(X_train)
        )


        print(
            "Testing rows:",
            len(X_test)
        )



        # ==================================================
        # MODEL
        # ==================================================

        model=XGBClassifier(

            objective="multi:softmax",

            num_class=len(labels),

            eval_metric="mlogloss",

            n_estimators=300,

            max_depth=8,

            learning_rate=0.05,

            n_jobs=-1,

            random_state=42

        )



        print(
            "Training XGBoost ICD..."
        )


        model.fit(
            X_train,
            y_train
        )



        pred=model.predict(
            X_test
        )



        acc=accuracy_score(
            y_test,
            pred
        )


        f1=f1_score(
            y_test,
            pred,
            average="weighted"
        )



        print(
            "\nAccuracy:",
            acc
        )


        print(
            "Weighted F1:",
            f1
        )


        print(
            classification_report(
                y_test,
                pred,
                zero_division=0
            )
        )



        # ==================================================
        # SAVE MODEL
        # ==================================================


        bundle={

            "model":model,

            "encoder":self.encoder,

            "features":self.features,

            "mapping":reverse_map

        }


        joblib.dump(
            bundle,
            MODEL_PATH
        )


        print(
            "Saved:",
            MODEL_PATH
        )




# ======================================================
# MAIN
# ======================================================

def main():

    trainer=ICDTrainer()

    trainer.run()



if __name__=="__main__":

    main()

# import yaml
# import joblib
# import pandas as pd
# import numpy as np

# from pathlib import Path

# from sklearn.metrics import (
#     accuracy_score,
#     f1_score,
#     classification_report
# )

# from scipy.spatial.distance import jensenshannon



# # ======================================================
# # CONFIG
# # ======================================================

# CONFIG_PATH = Path(
#     "/content/synthetic-population_/config/params.yaml"
# )


# with open(CONFIG_PATH) as f:
#     params = yaml.safe_load(f)



# # ======================================================
# # EVALUATION CLASS
# # ======================================================

# class EvaluateICD:


#     def __init__(self):

#         print("Starting ICD evaluation")


#         self.output_path = Path(
#             params["evaluate"]["output"][0]
#         )


#         self.report_path = (
#             self.output_path /
#             "reports"
#         )

#         self.report_path.mkdir(
#             parents=True,
#             exist_ok=True
#         )


#         files=params["evaluate"]["input"]



#         # -------------------------------
#         # Load datasets
#         # -------------------------------

#         print("\nLoading datasets")


#         self.synthetic=pd.read_csv(
#             files[0],
#             low_memory=False
#         )


#         self.test=pd.read_csv(
#             files[1],
#             low_memory=False
#         )


#         self.train=pd.read_csv(
#             files[2],
#             low_memory=False
#         )



#         print(
#             "Synthetic:",
#             self.synthetic.shape
#         )

#         print(
#             "Test:",
#             self.test.shape
#         )

#         print(
#             "Train:",
#             self.train.shape
#         )



#         # -------------------------------
#         # Load model
#         # -------------------------------

#         print("\nLoading model")


#         bundle=joblib.load(
#             files[3]
#         )


#         print(
#             "Bundle keys:",
#             bundle.keys()
#         )


#         self.model=bundle["model"]

#         self.features=bundle["features"]



#         # ==================================================
#         # Detect model format
#         # ==================================================

#         if "feature_encoders" in bundle:


#             print(
#                 "Using NEW training format"
#             )


#             self.feature_encoders = (
#                 bundle["feature_encoders"]
#             )


#             self.target_encoder = (
#                 bundle["target_encoder"]
#             )


#             self.old_format=False



#         elif "encoder" in bundle:


#             print(
#                 "Using OLD training format"
#             )


#             self.encoder=bundle["encoder"]

#             self.mapping=bundle.get(
#                 "mapping",
#                 {}
#             )

#             self.old_format=True



#         else:

#             raise ValueError(
#                 "Unknown model bundle format"
#             )





#     # ======================================================
#     # CLEAN
#     # Same as train
#     # ======================================================

#     def clean_data(self,df):

#         df=df.copy()


#         target="PRINC_DIAG_CODE"


#         df=df[
#             df[target].notna()
#         ]


#         df[target]=(
#             df[target]
#             .astype(str)
#             .str.strip()
#         )


#         df=df[
#             df[target]!="UNKNOWN"
#         ]



#         # AGE cleaning

#         df["PAT_AGE"]=(
#             df["PAT_AGE"]
#             .astype(str)
#             .str.replace(
#                 r"\D",
#                 "",
#                 regex=True
#             )
#         )


#         df["PAT_AGE"]=pd.to_numeric(
#             df["PAT_AGE"],
#             errors="coerce"
#         )


#         df=df[
#             df["PAT_AGE"].notna()
#         ]


#         return df





#     # ======================================================
#     # FEATURE ENCODING
#     # ======================================================

#     def encode_features(self,df):


#         df=df.copy()



#         # -------------------------------
#         # New format
#         # -------------------------------

#         if not self.old_format:


#             for col,encoder in self.feature_encoders.items():


#                 if col in df.columns:


#                     df[col]=(
#                         df[col]
#                         .astype(str)
#                     )


#                     known=set(
#                         encoder.classes_
#                     )


#                     df[col]=df[col].apply(
#                         lambda x:
#                         x if x in known
#                         else encoder.classes_[0]
#                     )


#                     df[col]=encoder.transform(
#                         df[col]
#                     )



#         # -------------------------------
#         # Old format
#         # -------------------------------

#         else:


#             cols=list(
#                 self.encoder.feature_names_in_
#             )


#             temp=df[cols].copy()


#             temp=temp.astype(str)


#             encoded=self.encoder.transform(
#                 temp
#             )


#             encoded=pd.DataFrame(
#                 encoded,
#                 columns=cols,
#                 index=df.index
#             )


#             for c in cols:

#                 df[c]=encoded[c]



#         return df




#     # ======================================================
#     # PREPROCESS
#     # ======================================================

#     def preprocess(self,df):


#         df=self.clean_data(
#             df
#         )


#         df=self.encode_features(
#             df
#         )



#         X=df[
#             self.features
#         ].copy()



#         for c in X.columns:

#             X[c]=pd.to_numeric(
#                 X[c],
#                 errors="coerce"
#             )



#         X=X.fillna(-1)



#         # target

#         if not self.old_format:


#             y=self.target_encoder.transform(
#                 df["PRINC_DIAG_CODE"]
#             )



#         else:


#             y=df["PRINC_DIAG_CODE"].map(
#                 self.mapping
#             )


#             valid=y.notna()


#             X=X.loc[valid]

#             y=y.loc[valid]


#             y=y.astype(int)



#         return X,y





#     # ======================================================
#     # DISTRIBUTION
#     # ======================================================

#     def js_similarity(self,a,b,name):


#         p=a.value_counts(
#             normalize=True
#         )


#         q=b.value_counts(
#             normalize=True
#         )


#         idx=p.index.union(
#             q.index
#         )


#         p=p.reindex(
#             idx,
#             fill_value=0
#         )


#         q=q.reindex(
#             idx,
#             fill_value=0
#         )


#         js=jensenshannon(
#             p.values,
#             q.values
#         )


#         score=(1-js)*100


#         print(
#             name,
#             ":",
#             round(score,2)
#         )


#         return score





#     def evaluate_distribution(self):


#         print(
#             "\nDistribution evaluation"
#         )


#         eval_df=pd.concat(
#             [
#                 self.synthetic,
#                 self.test
#             ],
#             ignore_index=True
#         )


#         results={}


#         for col in [

#             "PRINC_DIAG_CODE",
#             "SEX_CODE",
#             "RACE",
#             "ETHNICITY",
#             "APR_MDC"

#         ]:


#             results[col]=self.js_similarity(
#                 eval_df[col],
#                 self.train[col],
#                 col
#             )



#         pd.DataFrame(
#             {
#                 "Feature":results.keys(),
#                 "Similarity":results.values()
#             }
#         ).to_csv(
#             self.report_path /
#             "distribution_similarity.csv",
#             index=False
#         )





#     # ======================================================
#     # MODEL EVALUATION
#     # ======================================================

#     def evaluate_accuracy(self):


#         print(
#             "\nModel evaluation"
#         )



#         eval_df=pd.concat(
#             [
#                 self.synthetic,
#                 self.test
#             ],
#             ignore_index=True
#         )



#         X,y=self.preprocess(
#             eval_df
#         )


#         print(
#             "Evaluation X:",
#             X.shape
#         )


#         print(
#             "Classes:",
#             len(np.unique(y))
#         )



#         pred=self.model.predict(
#             X
#         )



#         acc=accuracy_score(
#             y,
#             pred
#         )


#         f1w=f1_score(
#             y,
#             pred,
#             average="weighted",
#             zero_division=0
#         )


#         f1m=f1_score(
#             y,
#             pred,
#             average="macro",
#             zero_division=0
#         )



#         print("\nAccuracy:",acc)

#         print(
#             "Weighted F1:",
#             f1w
#         )

#         print(
#             "Macro F1:",
#             f1m
#         )



#         report=classification_report(
#             y,
#             pred,
#             zero_division=0
#         )


#         print(report)



#         with open(
#             self.report_path /
#             "model_report.txt",
#             "w"
#         ) as f:


#             f.write(
#                 f"Accuracy: {acc}\n"
#             )

#             f.write(
#                 f"Weighted F1: {f1w}\n"
#             )

#             f.write(
#                 f"Macro F1: {f1m}\n\n"
#             )

#             f.write(
#                 report
#             )





#     # ======================================================
#     # RUN
#     # ======================================================

#     def run(self):

#         self.evaluate_distribution()

#         self.evaluate_accuracy()




# # ======================================================
# # MAIN
# # ======================================================

# def main():

#     evaluator=EvaluateICD()

#     evaluator.run()



# if __name__=="__main__":

#     main()
