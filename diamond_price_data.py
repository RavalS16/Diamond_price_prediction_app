import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error,r2_score
import pickle

st.set_page_config( 
    page_title="My Wide App",
    layout="wide",  # Options: "centered" (default) or "wide"
    initial_sidebar_state="expanded",
)

st.title("Diamond price prediction")

# data load
df = pd.read_csv("diamonds.csv")
df = df.drop(columns=["Unnamed: 0"])
num_col_eda = ['carat','depth','table','x','y','z','price']
cat_col_eda = ['cut','color','clarity']

b1 = st.sidebar.checkbox("show Dataset")
if b1:
    st.write(df.head())

#EDA
col1,col2 = st.columns([1,1])
b2 = st.sidebar.checkbox("EDA")
if b2:
    st.write("EDA")
    s1 = st.sidebar.selectbox("Type Of EDA",['univariate','bivariate','multivariate'])
    if s1 == 'univariate':
        col_name = st.sidebar.selectbox("select column",df.columns)
        if col_name in num_col_eda:
            chart1 = px.histogram(df[col_name])
            col1.plotly_chart(chart1)

            chart2 = px.box(df[col_name])
            col2.plotly_chart(chart2)
        elif col_name in cat_col_eda:
            df_cat_eda = df.groupby(col_name)['price'].count().reset_index()
            df_cat_eda.rename(columns={'price':'count'},inplace=True)
            #st.write(df_cat_eda)
            chart1 = px.bar(df_cat_eda,col_name,y='count')
            col1.plotly_chart(chart1)

            chart2 = px.pie(df_cat_eda,col_name,'count')
            col2.plotly_chart(chart2)
    elif s1 == 'bivariate':
        col_name = st.sidebar.selectbox("select column",df.columns)
        target_col = df['price']
        if col_name in num_col_eda:
            chart1 = px.scatter(df,col_name,target_col)
            col1.plotly_chart(chart1)
            df_num_eda = df.groupby(col_name)['price'].mean().reset_index()
            chart2 =px.line(df_num_eda,col_name,'price')
            col2.plotly_chart(chart2)
        pass
    else:
        pass


x = df.drop(columns=['price'])
y = df['price']

test_size = st.sidebar.slider("select test size",0.20,0.95,0.20,0.05)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_size,random_state=7)



b3 = st.sidebar.checkbox("check shape")
if b3 :
    st.write(f"x train {x_train.shape}")
    st.write(f"x test {x_test.shape}")
    st.write(f"y train {y_train.shape}")
    st.write(f"y test {y_test.shape}")


st.write(x_train)

num_train = x_train.drop(columns=['cut','color','clarity'])
num_test = x_test.drop(columns=['cut','color','clarity'])

cat_train = x_train[['cut','color','clarity']]
cat_test = x_test[['cut','color','clarity']]


scaler = StandardScaler()
num_train_s = scaler.fit_transform(num_train)
num_train_s = pd.DataFrame(num_train_s,columns = num_train.columns,
        index=num_train.index)


Encoder = OrdinalEncoder(categories=[['Fair','Good','Very Good','Premium','Ideal'],
                                     ['D','E','F','G','H','I','J'],
                                     ['I1','SI1','SI2','VS1','VS2','VVS1','VVS2','IF']])
cat_train_e = Encoder.fit_transform(cat_train)
cat_train_e = pd.DataFrame(cat_train_e,columns = cat_train.columns,
        index=cat_train.index)


x_train_new = num_train_s.join(cat_train_e)
x_train_new.head()


num_test_s = scaler.transform(num_test)
num_test_s = pd.DataFrame(num_test_s,columns = num_test.columns,
        index=num_test.index)


cat_test_e = Encoder.transform(cat_test)
cat_test_e = pd.DataFrame(cat_test_e,columns= cat_test.columns,
        index=cat_test.index)


x_test_new = num_test_s.join(cat_test_e)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=2.0),
    "Lasso": Lasso(alpha=0.5),
    "Decision Tree": DecisionTreeRegressor(random_state=7),
    "Random Forest": RandomForestRegressor(random_state=7),
    "Gradient Boosting": GradientBoostingRegressor(random_state=7),
    "AdaBoost": AdaBoostRegressor(random_state=7),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor()
}

st.sidebar.header("Model Training")

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

model = models[selected_model]

b4 = st.sidebar.button("Train Model")

if b4:

    model.fit(x_train_new, y_train)

    y_pred = model.predict(x_test_new)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.session_state['model'] = model
    st.session_state['y_pred'] = y_pred
    st.session_state['metrics'] = [mae, mse, rmse, r2]

    st.success("Model Trained Successfully!")


# Model Performance
st.header("Model Performance")

if 'metrics' in st.session_state:

    mae, mse, rmse, r2 = st.session_state['metrics']

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("MSE", f"{mse:.2f}")
    col3.metric("RMSE", f"{rmse:.2f}")
    col4.metric("R2 Score", f"{r2:.4f}")

    result_df = pd.DataFrame({
        'Actual Price': y_test.values,
        'Predicted Price': st.session_state['y_pred']
    })

    st.dataframe(result_df.head(20), use_container_width=True)

    chart = px.scatter(
        result_df,
        x='Actual Price',
        y='Predicted Price',
        title="Actual vs Predicted Price"
    )

    st.plotly_chart(chart, use_container_width=True)

else:

    st.info("Click Train Model to see performance.")


models = {'Pred_model':model,"Scaler":scaler,"Encoder":Encoder}
# Convert dictionary into pickle bytes
model_bytes = pickle.dumps(models)

# Download button
st.download_button(
        label="Download Models",
        data=model_bytes,
        file_name="models.pkl")