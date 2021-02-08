import pandas as pd
import numpy as np
from datetime import datetime
import scipy.optimize as optimize
import chart_studio.plotly as ply
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from dateutil.relativedelta import *


def present_value(r: float, coupon: float, number_of_payments,
                  days_till_next_payment, notional=100):
    """Assume 360 days per year;
    Assume r is given as annual rate;
    Assume days_till_next_payment <= 180"""
    r = r / 100
    pv = notional / ((1 + r * days_till_next_payment / 360) *
                     ((1 + r / 2) ** (number_of_payments - 1)))
    pmt = notional * (coupon / 100) / 2
    for i in range(number_of_payments):
        pv += pmt / ((1 + r * days_till_next_payment / 360) * ((1 + r /
                                                                2) ** i))
    return pv


def calculate_ytm(price: float, coupon: float, number_of_payments,
                  days_till_next_payment, notional=100):
    """Assume price is the already the dirty price(full price of the bond)
    Assume 360 days per year"""
    pmt = notional * (coupon / 100) / 2
    ytm_func = lambda x: sum([pmt /
                             ((1 + x * days_till_next_payment / 360) *
                              ((1 + x / 2) ** i))
                             for i in range(number_of_payments)]) + \
                        notional / ((1 + x * days_till_next_payment / 360) *
                            ((1 + x / 2) ** (number_of_payments - 1))) - price
    return round(optimize.newton(ytm_func, 0.05), 4)


def main():
    # import 10 bonds data
    data = pd.read_excel("/Users/wu/Desktop/apm466 a1/10 bonds.xlsx")
    maturity_list = data["Maturity"].tolist()
    purchase_date_list = data["Date"].tolist()
    number_of_payments_list = []
    pmt_date_list = data["Payment date"].tolist()
    coupon_list = data["Coupon"].tolist()
    price_list = data["Price"].tolist()
    x_axis = []
    ytm_dict = {}
    dirt_price_list = []
    days_till_payment_list = []
    days_to_maturity_list = []
    for i in range(len(maturity_list)):
        price = price_list[i]
        coupon = coupon_list[i]
        pmt_date = datetime.strptime(pmt_date_list[i], '%m/%d/%Y')
        mature = maturity_list[i]
        today = purchase_date_list[i]
        mature = datetime.strptime(mature, '%m/%d/%Y')
        days_to_maturity = (mature - today).days
        number_of_payments = days_to_maturity // 180 + 1
        num_day_till_pmt = (pmt_date - today).days
        number_of_payments_list.append(number_of_payments)
        accrued_interest = coupon * (180 - num_day_till_pmt) / 365
        dirt_price = price + accrued_interest
        dirt_price_list.append(dirt_price)
        days_till_payment_list.append(num_day_till_pmt)
        days_to_maturity_list.append(days_to_maturity)
        if number_of_payments != 1:
            ytm = calculate_ytm(dirt_price, coupon, number_of_payments,
                                    num_day_till_pmt)
        else:
            ytm = ((100 + coupon / 2) / dirt_price - 1) * 360 / num_day_till_pmt
        if today.date() in ytm_dict:
            ytm_dict[today.date()].append(ytm)
        else:
            ytm_dict[today.date()] = [ytm]
        if mature not in x_axis:
            x_axis.append(mature)
    data["number of payments"] = number_of_payments_list
    data["dirt price"] = dirt_price_list
    data["days to pmt"] = days_till_payment_list
    data["days to maturity"] = days_to_maturity_list
    indices = []
    for key in ytm_dict:
        indices.append(key)

    ytm_graph = go.Figure(go.Scatter(x=x_axis,
                                           y=ytm_dict[indices[0]],
                                           mode='lines+markers',
                                           name=str(indices[0]),
                                     line_shape="spline"
                                     ))
    for i in range(1, len(indices)):
        ytm_graph.add_trace(go.Scatter(x=x_axis,
                                           y=ytm_dict[indices[i]],
                                           mode='lines+markers',
                                           name=str(indices[i]),
                                       line_shape="spline"))
    ytm_graph.update_layout(
        title="Yield Curve",
        xaxis_title="Maturity",
        yaxis_title="YTM",
    )
    ytm_graph.show()

    data.to_excel("/Users/wu/Desktop/apm466 a1/10 bonds new.xlsx")

    # spot curve
    spot_data = []
    for k in range(10):
        bond = data[data["ID"] == 1].iloc[k]
        coupon = bond["Coupon"]
        today = bond["Date"]
        base_days_to_pmt = bond["days to pmt"]
        p = bond["dirt price"]
        r = - (np.log(p / (100 + coupon / 2)) / (base_days_to_pmt / 360))
        rate_dict = {base_days_to_pmt: r}
        for i in range(2, 12):
            bond = data[data["ID"] == i].iloc[k]
            coupon = bond["Coupon"]
            pmt_date = datetime.strptime(bond["Payment date"], '%m/%d/%Y')
            p = bond["dirt price"]
            num_pmt = bond["number of payments"]
            curr = 0
            days_to_maturity = bond["days to maturity"]
            days = list(rate_dict.keys())
            for j in range(num_pmt - 1):
                date_temp = pmt_date + relativedelta(months=+(j*6))
                day_temp = (date_temp - today).days
                if day_temp not in days:
                    x = (np.array(days)).reshape((len(days), 1))
                    temp = np.zeros((len(days), 2))
                    temp[:, :1] = x
                    temp[:, 1:2] = x ** 2
                    rates = list(rate_dict.values())
                    y = (np.array(rates)).reshape((len(rates), 1))
                    reg = LinearRegression().fit(temp, y)
                    predictor = np.array([[day_temp, day_temp ** 2]])

                    rate = np.sum(reg.predict(predictor))
                else:
                    rate = rate_dict[days[j]]
                pv = (coupon / 2) * np.exp(-rate * days[j] / 360)
                curr += pv
            r_new = -(np.log((p - curr) / (100 + coupon / 2)) /
                      (days_to_maturity / 360))
            rate_dict[days_to_maturity] = r_new
        x = np.array(list(rate_dict.keys())) / 365
        y = np.array(list(rate_dict.values()))
        name = str(today)[: str(today).find(" ")]
        if k == 0:
            spot_curve = go.Figure(go.Scatter(x=x,
                                           y=y,
                                           mode='lines+markers',
                                           name=name,
                                     line_shape="spline"))
        else:
            spot_curve.add_trace(go.Scatter(x=x,
                                           y=y,
                                           mode='lines+markers',
                                           name=name,
                                           line_shape="spline"))
        spot_data.append(rate_dict)
    spot_curve.update_layout(
        title="Spot Curve",
        xaxis_title="Year",
        yaxis_title="Rate",
    )
    spot_curve.show()

    degree = 5

    forward_rates_data = []
    for k in range(10):
        bond = data[data["ID"] == 1].iloc[k]
        today = bond["Date"]
        rates_data = spot_data[k]
        rates = list(rates_data.values())
        term = list(rates_data.keys())
        x = (np.array(term)).reshape((len(term), 1))
        temp = np.zeros((len(term), degree))
        temp[:, 0:1] = x
        for i in range(1, degree):
            temp[:, i: i + 1] = x ** (i + 1)
        y = (np.array(rates)).reshape((len(rates), 1))
        reg = LinearRegression().fit(temp, y)
        base_date = today + relativedelta(months=+12)
        base_term = (base_date - today).days
        predictor = [base_term]
        for order in range(1, degree):
            predictor.append(base_term ** (order + 1))
        base_rate = np.sum(reg.predict(np.array([predictor])))
        forward_rates = {}
        subtractor = base_rate * base_term
        for j in range(2, 6):
            date = today + relativedelta(months=+(12 * j))
            term = (date - today).days
            predictor = [term]
            for order in range(1, degree):
                predictor.append(term ** (order + 1))
            spot_rate = np.sum(reg.predict(np.array([predictor])))
            new_rate = (spot_rate * term - subtractor) / (term - base_term)
            forward_rates[term - 365] = new_rate
        for m in rates_data:
            if m > 366:
                term = m
                spot_rate = rates_data[m]
                new_rate = (spot_rate * term - subtractor) / (term - base_term)
                forward_rates[term - 365] = new_rate
        forward_rates = dict(sorted(forward_rates.items()))
        x = np.array(list(forward_rates.keys())) / 365
        y = np.array(list(forward_rates.values()))
        name = str(today)[: str(today).find(" ")]
        if k == 0:
            forward_curve = go.Figure(go.Scatter(x=x,
                                              y=y,
                                              mode='lines+markers',
                                              name=name,
                                              line_shape="spline"))
        else:
            forward_curve.add_trace(go.Scatter(x=x,
                                            y=y,
                                            mode='lines+markers',
                                            name=name,
                                            line_shape="spline"))
        forward_rates_data.append(forward_rates)
    forward_curve.update_layout(
        title="1 year Forward Curve",
        xaxis_title="Year",
        yaxis_title="Rate",
    )
    forward_curve.show()
    num_years = 6

    # prepare ytm and forward data matrices
    ytm_data = {1: [], 2: [], 3: [], 4: [], 5: []}
    forward_data = {1: [], 2: [], 3: [], 4: []}
    for k in range(10):
        x = []
        x1 = list(forward_rates_data[k].keys())
        for i in range(1, 12):
            bond = data[data["ID"] == i].iloc[k]
            x.append(bond["days to maturity"])
        y = ytm_dict[indices[k]]
        y1 = list(forward_rates_data[k].values())
        x = (np.array(x)).reshape((len(x), 1))
        x1 = (np.array(x1)).reshape((len(x1), 1))
        temp = np.zeros((len(x), degree))
        temp1 = np.zeros((len(x1), degree))
        temp[:, 0:1] = x
        temp1[:, 0:1] = x1
        for i in range(1, degree):
            temp[:, i: i + 1] = x ** (i + 1)
            temp1[:, i: i + 1] = x1 ** (i + 1)
        y = (np.array(y)).reshape((len(y), 1))
        y1 = (np.array(y1)).reshape((len(y1), 1))
        reg = LinearRegression().fit(temp, y)
        reg1 = LinearRegression().fit(temp1, y1)
        for year in range(1, num_years):
            predictor = [year * 365]
            for order in range(1, degree):
                predictor.append((year * 365) ** (order + 1))
            ytm_prediction = np.sum(reg.predict(np.array([predictor])))
            ytm_data[year].append(ytm_prediction)
        for year in range(1, num_years - 1):
            predictor = [year * 365]
            for order in range(1, degree):
                predictor.append((year * 365) ** (order + 1))
            forward_prediction = np.sum(reg1.predict(np.array([predictor])))
            forward_data[year].append(forward_prediction)
    ytm_dat = {1: [], 2: [], 3: [], 4: [], 5: []}
    forward_dat = {1: [], 2: [], 3: [], 4: []}
    for i in range(1, num_years):
        for j in range(1, len(ytm_data[i])):
            ytm_dat[i].append(np.log(ytm_data[i][j] /
                                     ytm_data[i][j - 1]))
    for i in range(1, num_years - 1):
        for j in range(1, len(forward_data[i])):
            forward_dat[i].append(np.log(forward_data[i][j] /
                                     forward_data[i][j - 1]))
    ytm_mat = np.zeros((len(ytm_dat[1]), num_years - 1))
    forward_mat = np.zeros((len(forward_dat[1]), num_years - 2))
    for i in range(1, num_years):
        ytm_mat[:, i-1:i] = \
            (np.array(ytm_dat[i])).reshape((len(ytm_dat[1]), 1))
    for i in range(1, num_years - 1):
        forward_mat[:, i-1:i] = \
            (np.array(forward_dat[i])).reshape((len(forward_dat[1]), 1))
    ytm_cov = np.cov(ytm_mat.T)
    forward_cov = np.cov(forward_mat.T)


if __name__ == "__main__":
    main()


