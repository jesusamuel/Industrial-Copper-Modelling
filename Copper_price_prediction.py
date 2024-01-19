import pandas as pd
import streamlit as st
import mysql.connector
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,mean_squared_error,r2_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

mydb = mysql.connector.connect(host='localhost',
                              user='root',
                              password='Jesu@123',
                              database='Copper')

mycursor = mydb.cursor()

sql="select * from copper_data;"
mycursor.execute(sql)  
data=mycursor.fetchall()
sql="DESC copper_data;"
mycursor.execute(sql)  
data1=mycursor.fetchall()
ad=list(pd.DataFrame(data1)[0]) 
df=pd.DataFrame(data=data,columns=ad)
res=df.head()

##############################################################################################################################
def Selling_price():
    qty=st.number_input("Enter the Quantity in tons:")
    cry=st.number_input("Enter the Country:")
    app=st.number_input("Enter the Application number:")
    tks=st.number_input("Enter the Thickness:")
    wdt=st.number_input("Enter the width:")
    pdt=st.number_input("Enter the Product_ref code:")
    sts=st.number_input("Enter the Status[W/L as 1/0]:")
    if st.button("Submit"):
        TT={
            "quantity_tons":[qty,],
            "country":[cry,],
            "application":[app,],
            "thickness":[tks,],
            "width":[wdt,],
            "product_ref":[pdt,],
            "T_status":[sts,]
        }

        t1=pd.DataFrame(TT)

        X=df.drop(['id','item_date','status','customer','item_type','material_ref','delivery_date','selling_price'],axis=1)
        y=df['selling_price']

        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
        best_algo=['name',0] #####
        models=[LinearRegression(),]
        for model in models:
            model.fit(x_train,y_train)

            train_pred=model.predict(x_train)
            test_pred=model.predict(x_test)
            
            predicted_value =model.predict(t1)##########
            
            st.write("Predicted Value: ",predicted_value)

            st.write('\n\n\n****Train****')
            st.write(f'Mean Square Error:{mean_squared_error(y_train,train_pred)}')
            st.write(f'R2 Score:{r2_score(y_train,train_pred)}')
            
            st.write('****Test****')

            st.write(f'Mean Square Error:{mean_squared_error(y_test,test_pred)}')
            st.write(f'R2 Score:{r2_score(y_test,test_pred)}')

            
            st.write("\n")
        
###############################################################################################################################
def Status():    
        
    qty1=st.number_input("Enter the Quantity in tons:")
    cry1=st.number_input("Enter the Country:")
    app1=st.number_input("Enter the Application number:")
    tks1=st.number_input("Enter the Thickness:")
    wdt1=st.number_input("Enter the width:")
    pdt1=st.number_input("Enter the Product_ref code:")
    prs1=st.number_input("Enter the Selling Price:")    
    if st.button("Submit"):

        TT={
            "quantity_tons":[qty1,],
            "country":[cry1,],
            "application":[app1,],
            "thickness":[tks1,],
            "width":[wdt1,],
            "product_ref":[pdt1,],
            "selling_price":[prs1,]
        }

        t1=pd.DataFrame(TT)
        t1
        
        model=LabelEncoder()
        df['T_status']=model.fit_transform(df['status'])

        X=df.drop(['id','item_date','status','customer','item_type','material_ref','delivery_date','T_status'],axis=1)
        y=df['T_status']

        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
        best_algo=['name',0] #####
        models=[LogisticRegression(),KNeighborsClassifier(),RandomForestClassifier()]
        for model in models:
            model.fit(x_train,y_train)

            train_pred=model.predict(x_train)
            test_pred=model.predict(x_test)

            st.write('****Train****')
            st.write(f'Training Accuracy:{accuracy_score(y_train,train_pred)}')
            st.write(f'Training Precicsion:{precision_score(y_train,train_pred)}')
            st.write(f'Training Recall:{recall_score(y_train,train_pred)}')
            st.write(f'Training F1_score:{f1_score(y_train,train_pred)}')
            st.write('****Test****')

            st.write(f'Test Accuracy:{accuracy_score(y_test,test_pred)}')
            st.write(f'Test Precicsion:{precision_score(y_test,test_pred)}')
            st.write(f'Test Recall:{recall_score(y_test,test_pred)}')
            st.write(f'Test F1_score:{f1_score(y_test,test_pred)}')
            
            st.write("\n")
            
            if best_algo[1] <= f1_score(y_test,test_pred):
                best_algo=[type(model).__name__,f1_score(y_test,test_pred)]
                
        st.write("Best Algorithm:",best_algo)   

        str1=[]
        str1.append(best_algo[0])
        str1.append('()')
        model=''.join(str1)
        model=eval(model)
        model.fit(x_train,y_train)
        res=model.predict(t1)
    model_name1 =st.sidebar.selectbox(
            "Select the option to check values:",
            ("Training / Test Prediction", "Best Algorithm"),key=1,index=None
            )
    if model_name1 == 'Training / Test Prediction':
        model.fit(x_train,y_train)

        train_pred=model.predict(x_train)
        test_pred=model.predict(x_test)

        st.write('****Train****')
        st.write(f'Training Accuracy:{accuracy_score(y_train,train_pred)}')
        st.write(f'Training Precicsion:{precision_score(y_train,train_pred)}')
        st.write(f'Training Recall:{recall_score(y_train,train_pred)}')
        st.write(f'Training F1_score:{f1_score(y_train,train_pred)}')
        st.write('****Test****')

        st.write(f'Test Accuracy:{accuracy_score(y_test,test_pred)}')
        st.write(f'Test Precicsion:{precision_score(y_test,test_pred)}')
        st.write(f'Test Recall:{recall_score(y_test,test_pred)}')
        st.write(f'Test F1_score:{f1_score(y_test,test_pred)}')
            
        st.write("\n")
        st.write("Predicted Value:",str(res[0]))
        


###########################################################################################################

def main():

    st.header('                  Industrial Copper Modelling', divider='rainbow')

    #model_name =st.selectbox('',('Predict price of copper','Predict W/L status of copper'),index=None,placeholder="Choose an option to predict")  

    model_name = st.sidebar.selectbox(
    "Choose the option which you want to predict",
    ("Home", 'Predict price of copper', 'Predict W/L status of copper'),index=0
    )
    

    if model_name == 'Predict price of copper' :
        Selling_price()
    elif model_name == 'Predict W/L status of copper' :
        Status()  
    else:
        st.write("""  The Industrial copper modelling project is used to predict
                 the Continuous value like Selling Price and Categorical value like
                 Won or Lost. Based on the provided dataset, we have pre-processed the data.
                 handling null values and encoding the categorical columns.
                 We have used Continous model to be as Linear Regression as it is best fit
                 for given dataset. And Classification model such as RandomForestClassifier,
                 LogisticRegression,KNeighbourClassifier""")     
        st.write("\n\n\n")
        st.bar_chart(df,x='country',y='selling_price')




if __name__ == "__main__":
    main()
        
