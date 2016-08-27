# -*- coding: utf-8 -*-
import oandapy, matplotlib.pyplot as plt, sqlite3, csv, argparse, locale, requests, threading, json, arrow, thread, decimal, math,  numpy as np
from pprint import pprint
from time import sleep, time
from decimal import Decimal
from xml.dom import minidom
from datetime import datetime, timedelta
from collections import deque
from HTMLParser import HTMLParser

def main():
    locale.setlocale(locale.LC_ALL, '')
    parser = argparse.ArgumentParser(
            description='xauforex: use gold price change for forex on oanda')
    parser.add_argument('-s', '--save', type=str,
                        help='Save the price to <SAVE>', required = True)
    parser.add_argument('-c', '--config', type=str,
                        help='Config file location <FILE>', required = True)
    subparsers = parser.add_subparsers()

    parser_status   = subparsers.add_parser('status',
            help='show account status')
    parser_status.set_defaults(func = main_status)
    parser_status.add_argument('--interval', type=int, default = 0,
            help='interval for displaying status')

    parser_draw    = subparsers.add_parser('draw',   help='draw graph')
    parser_draw.set_defaults(func = main_draw)
    parser_draw.add_argument('--sma', default = False, 
            help='draw sma curve')
    parser_draw.add_argument('--trend', action='store_true',
            help='draw trend line')
    parser_draw.add_argument('-f', '--from-time', type=int,
            help='draw from <FROM> in unix timestamp')
    parser_draw.add_argument('-t', '--to-time', type=int,
            help='draw to <TO> in unix timestamp')

    parser_monitor = subparsers.add_parser('monitor',
            help='monitor price change and save to db')
    parser_monitor.set_defaults(func = main_monitor)
    parser_monitor.add_argument('-v', '--verbose', action='store_true',
            help='display verbose output')

    parser_trade   = subparsers.add_parser('trade', 
            help='start trading at oanda')
    parser_trade.add_argument('--expire-diff', type=int, default = 24*60*60,
            help='seconds until order expires')
    parser_trade.add_argument('--base-units', type=int, default = 10,
            help='order base units')
    parser_trade.add_argument('--trailing-stop', type=int, default = 0,
            help='trailing stop pips')
    parser_trade.add_argument('-v', '--verbose',
            action = 'store_true', default = False,
            help='verbose output')
    parser_trade.add_argument('--interval', type=int, default = 5,
            help='interval for checking order')
    parser_trade.set_defaults(func = main_trade)

    parser_monitortrade = subparsers.add_parser('monitortrade',
            help='monitor and price change and save to db, and trade')
    parser_monitortrade.set_defaults(func = main_monitortrade)
    parser_monitortrade.add_argument('-v', '--verbose', action='store_true',
            help='display verbose output')
    parser_monitortrade.add_argument('--expire-diff', type=int, default = 24*60*60,
            help='seconds until order expires')
    parser_monitortrade.add_argument('--base-units', type=int, default = 10,
            help='order base units')
    parser_monitortrade.add_argument('--trailing-stop', type=int, default = 0,
            help='trailing stop pips')
    parser_monitortrade.add_argument('--interval', type=int, default = 5,
            help='interval for checking order')
    parser_monitortrade.set_defaults(func = main_monitortrade)

    parser_backtest   = subparsers.add_parser('backtest',
            help='exec backtest using db')
    parser_backtest.set_defaults(func = main_backtest)
    parser_backtest.add_argument('-f', '--from-time', type=int, required=True,
            help='backtest from <FROM> in unix timestamp')
    parser_backtest.add_argument('-t', '--to-time', type=int, required=True,
            help='backtest to <TO> in unix timestamp')
    parser_backtest.add_argument('--expire-diff', type=int, default = 24*60*60,
            help='seconds until order expires')
    parser_backtest.add_argument('--base-units', type=int, default = 10,
            help='order base units')
    parser_backtest.add_argument('--trailing-stop', type=int, default = 0,
            help='trailing stop pips')
    parser_backtest.add_argument('-v', '--verbose',
            action = 'store_true', default = False,
            help='verbose output')
    parser_backtest.add_argument('--backtest-time-diff',
            type=int, default = 5,
            help='time diff to exec backtest')
    parser_backtest.add_argument('--gain-graph',
            action = 'store_true',
            help='draw gain graph')


    args = parser.parse_args()
    if args.config:
        config_file = args.config
    else:
        config_file = 'config.json'
    with open(config_file) as data_file:
        args.config = json.load(data_file)

    args.prices = PriceStore(args.save)
    args.orders = OrderStore(args.save)

    args.func(args)

def main_monitortrade(args):
    try:
        thread.start_new_thread(main_monitor, (args,))
        thread.start_new_thread(main_trade, (args,))
        while 1:
            sleep(1)
            pass
    except (KeyboardInterrupt):
        print ("aborted")


def main_trade(args):
    access_token = args.config['oanda']['access_token']
    account_id   = args.config['oanda']['account_id']
    oanda = oandapy.API(environment="practice",\
            access_token=access_token)

    def prec(num):
        return "{:.2f}".format(num)
    def time_format(time_):
        return datetime.utcfromtimestamp(
            time_).isoformat("T") + "Z"
    def tradable_day(current_time):
        """
            return True iff market is open
            tocom : 月曜9:00-土曜4:00ま(東京時間)
                      GMT+0だと月曜0:00 - 金曜19:00
            lbma  : The auction runs twice daily at 10:30am
                        and 3:00pm London time.
                      月曜10:30-金曜15:00 - 金曜24:00?
            したがって月曜10:30 - 金曜19:00(GMT+0)が共通の取引時間
        """
        utc = datetime.utcfromtimestamp(current_time)
        # utc = datetime.now()
        weekday = utc.weekday()
        minute = utc.minute + utc.hour * 60
        # skip holidays
        if(weekday >= 5):
            return False
        # skip early time in monday
        elif(weekday == 0 and minute < 11*60 + 30):
            return False
        # skip late time in friday
        elif(weekday == 4 and minute > 19*60):
            return False
        else:
            return True
 
    prices = PriceStore(args.save)
    orders = OrderStore(args.save)
    # prices = args.prices
    # orders = args.orders
    base_units = args.base_units
    expire_diff = args.expire_diff

    order_type = 'limit'
    trailingStop = args.trailing_stop
    interval = args.interval
    verbose  = args.verbose
    lower_bound_rate = Decimal(1-1e-4)
    upper_bound_rate = Decimal(1+1e-4)

    try:
        while True:
            current_time =  int(datetime.now().strftime("%s"))
            status, order_at, settle_at, stop_loss \
                    = buy_sell_stay(prices, current_time)

            if not tradable_day(current_time):
                """
                    NO_OP
                """

            elif (status == 'ignore' or status == 'stay'):
                if(verbose):
                    print status, order_at, settle_at, stop_loss
            else:
                # create order if not exist
                # only when network is available
                if(verbose):
                    print status, order_at, settle_at, stop_loss
                try:
                    lowerBound   = prec(order_at*lower_bound_rate)
                    upperBound   = prec(order_at*upper_bound_rate)
                    stopLoss     = prec(stop_loss)
                    takeProfit   = prec(settle_at)
                    price        = prec(order_at)

                    if (status == 'buy' and \
                            stopLoss < lowerBound and \
                            lowerBound < price and \
                            price < upperBound and \
                            upperBound < takeProfit \
                            ) or (status == 'sell' and \
                            takeProfit < lowerBound and \
                            lowerBound < price and \
                            price < upperBound and \
                            upperBound < stopLoss ):

                        response = oanda.create_order(account_id,
                           lowerBound   = lowerBound,
                           upperBound   = upperBound,
                           stopLoss     = stopLoss  ,
                           takeProfit   = takeProfit,
                           price        = price     ,
                           side         = status,
                           trailingStop = trailingStop,
                           instrument   = 'GBP_JPY',
                           units        = base_units,
                           expiry       = time_format(current_time \
                                            + expire_diff),
                           type         = order_type)
                        if(verbose):
                            print response
                except Exception as e:
                    print e
                    pass
            sleep(interval)
    
    except (KeyboardInterrupt):
        print ("aborted")

def main_backtest(args):
    def prec(num):
        return "{:.2f}".format(num)
    def time_format(time_):
        return datetime.utcfromtimestamp(
            time_).isoformat("T") + "Z"
    
    def tradable_day(current_time):
        """
            return True iff market is open
            tocom : 月曜9:00-土曜4:00ま(東京時間)
                      GMT+0だと月曜0:00 - 金曜19:00
            lbma  : The auction runs twice daily at 10:30am and 3:00pm London time.
                      月曜10:30-金曜15:00 - 金曜24:00?
            したがって月曜10:30 - 金曜19:00(GMT+0)が共通の取引時間
        """
        utc = datetime.utcfromtimestamp(current_time)
        # utc = datetime.now()
        weekday = utc.weekday()
        minute = utc.minute + utc.hour * 60
        # skip holidays
        if(weekday >= 5):
            return False
        # skip early time in monday
        elif(weekday == 0 and minute < 11*60 + 30):
            return False
        # skip late time in friday
        elif(weekday == 4 and minute > 19*60):
            return False
        else:
            return True
    
    prices = args.prices
    orders = args.orders
    orders.drop()
    base_units = args.base_units
    expire_diff = args.expire_diff
    order_type = 'limit'
    trailingStop = args.trailing_stop
    gain_graph = args.gain_graph
    lower_bound_rate = Decimal(1-1e-4)
    upper_bound_rate = Decimal(1+1e-4)
    backtest_time_diff = args.backtest_time_diff
    current_time = int(args.from_time)
    start_time = current_time
    until = int(args.to_time)
    try:
        while current_time < until:
            if not tradable_day(current_time):
                current_time += backtest_time_diff
                pass
            # order every 5 seconds
            current_time += backtest_time_diff
            # total_old = total
            status, order_at, settle_at, stop_loss \
                    = buy_sell_stay(prices, current_time)
            current_orders = orders.load()
            if (status == 'ignore' or status == 'stay'):
                pass
            else:
                assert (status == 'buy' and \
                        stop_loss < order_at and \
                        order_at < settle_at ) or \
                       (status == 'sell' and \
                        stop_loss > order_at and \
                        order_at > settle_at ), 'inconsistent order'
                # create order if not exist
                # only when network is available
                if(status == 'buy'):
                    print status, \
                        prec(stop_loss), \
                        prec(order_at*lower_bound_rate), \
                        prec(order_at), \
                        prec(order_at*upper_bound_rate), \
                        prec(settle_at)
                else:
                    print status, \
                        prec(settle_at), \
                        prec(order_at*lower_bound_rate), \
                        prec(order_at), \
                        prec(order_at*upper_bound_rate), \
                        prec(stop_loss)

                try: 
                    o = {
                      'lowerBound':   prec(order_at*lower_bound_rate),
                      'upperBound':   prec(order_at*upper_bound_rate),
                      'stopLoss':     prec(stop_loss),
                      'takeProfit':   prec(settle_at),
                      'price':        prec(order_at),
                      'side':         status,
                      'trailingStop': trailingStop,
                      'instrument':   'GBP_JPY',
                      'time':         current_time,
                      'units':        base_units,
                      'expiry':       current_time+expire_diff,
                      'type':         order_type,
                      'status':       'created',
                      'id':           current_time
                    }
                    if (status == 'buy' and \
                            o['stopLoss'] < o['lowerBound'] and \
                            o['lowerBound'] < o['price'] and \
                            o['price'] < o['upperBound'] and \
                            o['upperBound'] < o['takeProfit'] \
                            ) or (status == 'sell' and \
                            o['takeProfit'] < o['lowerBound'] and \
                            o['lowerBound'] < o['price'] and \
                            o['price'] < o['upperBound'] and \
                            o['upperBound'] < o['stopLoss'] ):
                        print o
                        orders.save(Order(o))

                except Exception as e:
                    print e
                    pass
        """
            this part of code is only for testing
            get orders from order_data table,
            get prices from price_data table
            for each price sorted by timestamp, if order can be closed,
            close order, and if stoploss should be applied, do so
        
        """
        x_oand, y_oand = get_time_series('OANDA', \
                    'GBP', 'JPY', start_time, until, prices)
        x_oand, y_oand_ask = get_time_series('OANDA', \
                'GBP', 'JPY', start_time, until, prices, func = lambda x: [x.get_ask(), x.timestamp])
        x_oand, y_oand_bid = get_time_series('OANDA', \
                'GBP', 'JPY', start_time, until, prices, func = lambda x: [x.get_bid(), x.timestamp])
        all_orders = orders.load(ts_start = start_time, ts_end = until)
        
        time, gain = [], []
        
        total = Decimal(0)
        for i in xrange(len(x_oand)):
            total_old = total
            current_time =  x_oand[i]
            current_price = Decimal(y_oand[i])
            current_ask = Decimal(y_oand_ask[i])
            current_bid = Decimal(y_oand_bid[i])
            for j in xrange(len(all_orders)):
                # all_orders[j] = all_orders[j]
                if(all_orders[j].time < current_time and \
                        all_orders[j].status == 'created'):
                    if(all_orders[j].side == 'buy' and \
                            all_orders[j].takeProfit < current_ask):
                        all_orders[j].status = 'buy profit'
                        # take profit
                        profit = (current_ask-Decimal(all_orders[j].price))\
                                *Decimal(all_orders[j].units)
                        total +=  profit
                        # print profit
                    elif(all_orders[j].side == 'buy' and \
                            all_orders[j].stopLoss > current_bid):
                        all_orders[j].status = 'buy stoploss'
                        # stop loss
                        profit = (current_bid-Decimal(all_orders[j].price))\
                                *Decimal(all_orders[j].units)
                        total +=  profit
                        # print profit
                    elif(all_orders[j].side == 'buy' and \
                            all_orders[j].expiry < current_time):
                        all_orders[j].status = 'buy expire'
                        profit = (current_ask-Decimal(all_orders[j].price))\
                                *Decimal(all_orders[j].units)
                        total +=  profit
                        # print profit
                    elif(all_orders[j].side == 'sell' and \
                            all_orders[j].takeProfit > current_bid):
                        all_orders[j].status = 'sell profit'
                        # take profit
                        profit = (-current_bid+Decimal(all_orders[j].price))\
                                *Decimal(all_orders[j].units)
                        total +=  profit
                        # print profit
                    elif(all_orders[j].side == 'sell' and \
                            all_orders[j].stopLoss < current_ask):
                        all_orders[j].status = 'sell stoploss'
                        # stop loss
                        profit = (-current_ask+Decimal(all_orders[j].price))\
                                *Decimal(all_orders[j].units)
                        total +=  profit
                    elif(all_orders[j].side == 'sell' and \
                            all_orders[j].expiry < current_time):
                        all_orders[j].status = 'sell expire'
                        profit = (-current_bid+Decimal(all_orders[j].price))\
                                *Decimal(all_orders[j].units)
                        total +=  profit
                        # print profit
        
                # elif(all_orders[j].expiry < current_time):
                #     all_orders[j].status = 'counted'
        
                if all_orders[j].status != 'created':
                    pass
                    # print all_orders[j].status
            if total != total_old:
                time.append(current_time)
                gain.append(total)
                print current_time, total
        
        if gain_graph:
            plt.plot(time, gain, "y")
            plt.show()
    except (KeyboardInterrupt):
        print ("aborted")

def main_status(args):
    access_token = args.config['oanda']['access_token']
    account_id = args.config['oanda']['account_id']
    oanda = oandapy.API(environment="practice",\
            access_token=access_token)
    # pprint (oanda.get_transaction_history(account_id)['transactions'][0])
    # pprint (oanda.get_eco_calendar(instrument = 'GBP_JPY', period = 604800))
    def time2int(x):
        try:
            _time = int(datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.000000Z').strftime("%s"))
        except:
            _time = int(datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ').strftime("%s"))

        return _time

    def history2xy(x):
        accountBalance = float(x['accountBalance'])
        time_ = time2int(x['time'])
        return np.array([time_,accountBalance])

    def history_has_balance_time(x):
        return ('accountBalance' in x) and ('time' in x) \
                and (time2int(x['time']) > 1471002929)

    try: 
        while True:
            print ("latest transaction history")
            history = oanda.get_transaction_history(account_id)['transactions']
            pprint (history[len(history)-1])
            print ("unconfirmed positions")
            pprint (oanda.get_positions(account_id))
            print ("orders list")
            print (oanda.get_orders(account_id))
            print ("account info")
            pprint (oanda.get_account(account_id))

            if not args.interval:
                break
            else:
                sleep(args.interval)
    except KeyboardInterrupt:
        print ("aborted")
    """
    history_data = []
    max_id = False
    while True:
        if(max_id):
            history = oanda.get_transaction_history(account_id, maxId = max_id)['transactions']
        else:
            history = oanda.get_transaction_history(account_id)['transactions']
            
        if(len(history)):
            history_data = history_data + history
            ids = map(lambda x: x['id'], history)
            max_id = min(ids) - 1
            print max_id
        else:
            break
    # curl -X GET -I -H "Authorization: Bearer 0000000000000000000-aaaaaaaaaaaaaaaaaaaaaa" \
    #        "https://api-fxpractice.oanda.com/v1/accounts/12345/alltransactions"
    with open("data/a.json") as data_file:
        history_data = json.load(data_file)
    xy = np.array(map(history2xy, filter(history_has_balance_time, history_data))).T
    plt.plot(xy[0], xy[1], "y")
    plt.show()
    """

def main_draw(args):
    if args.sma:
        sma_average = args.sma
    else:
        sma_average = False
    from_time = args.from_time
    to_time = args.to_time
    trend = args.trend
    store = args.prices
    show_graph(store, from_time=from_time, to_time=to_time, sma_average=sma_average, trend=trend)

def main_monitor(args):
    # called by thread
    def call_monitor(monitor, interval, filename, verbose):
        store = PriceStore(filename)
        e = threading.Event()
        while not e.wait(interval):
            if crawlable_day():
                p = monitor.price
                if p:
                    store.save(p)
                    if verbose:
                        print (vars(p))
    
    config = args.config
    
    xaugbp_bv = BullionVaultMonitor(
                username = config["bullionvault"]["username"],
                password = config["bullionvault"]["password"],
                currency='GBP',
                market = 'AUXLN',
                url="https://live.bullionvault.com/secure/api/v2/view_market_xml.do")
    xaujpy_bv = BullionVaultMonitor(
                username = config["bullionvault"]["username"],
                password = config["bullionvault"]["password"],
                currency='JPY',
                market = 'AUXZU',
                url="https://live.bullionvault.com/secure/api/v2/view_market_xml.do")
    xaujpy_tocom = TocomMonitor(
                currency='JPY',
                market = 'TOCOM',
                url = "http://www.tocom.or.jp/data/souba_tgs/{0}")
    xaujpy_xmlcharts = XmlChartsMonitor(
                 url="https://www.xmlcharts.com/cache/precious-metals.php?format=json&currency={0}",
                 currency="JPY",
                 market="XMLCHARTS.COM")
    xaugbp_xmlcharts = XmlChartsMonitor(
                 url="https://www.xmlcharts.com/cache/precious-metals.php?format=json&currency={0}",
                 currency="GBP",
                 market="XMLCHARTS.COM")
    xaugbp_quandl = QuandlMonitor(currency='GBP',
                api_key = config['quandl']['api_key'],
                url="https://www.quandl.com/api/v3/datasets/LBMA/GOLD.json?start_date={0}",
                market="QUANDL.COM")
    gbpjpy_oanda = OandaMonitor(count = 0,
            filename = args.save,
            environment = 'practice',
            access_token = config["oanda"]["access_token"],
            account_id = config["oanda"]["account_id"])
    event_calendar = EventCalendarMonitor(
            filename = args.save,
            access_token = config["oanda"]["access_token"],
            environment = 'practice',
            period = 604800,
            instrument = 'GBP_JPY',
            duration = 60)
    
    # updated occasionary
    thread.start_new_thread(call_monitor, (xaugbp_bv, 60, args.save, args.verbose))
    thread.start_new_thread(call_monitor, (xaujpy_bv, 60, args.save, args.verbose))
    # updated 5 min
    thread.start_new_thread(call_monitor, (xaujpy_tocom, 60, args.save, args.verbose))
    # limited request : 48 / day
    thread.start_new_thread(call_monitor, (xaujpy_xmlcharts, 1800, args.save, args.verbose))
    thread.start_new_thread(call_monitor, (xaugbp_xmlcharts, 1800, args.save, args.verbose))
    # limited request : 50 / day
    thread.start_new_thread(call_monitor, (xaugbp_quandl, 1728, args.save, args.verbose))
    # use streaming api
    thread.start_new_thread(gbpjpy_oanda.start_monitor, (args.verbose,))
    # thread.start_new_thread(event_calendar.start_monitor, ())
    
    try:
        while 1:
            sleep(1)
            pass
    except (KeyboardInterrupt):
        print ("aborted")

"""
    線形補間
"""
def lerp(p1, p2, x):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    return (y2 - y1)/((x2 - x1)*Decimal(1.0)) * (x - x1) + y1

"""
    (x2,y2)上におけるx1の値を補間して返す
    x1 and y1 have same length
    x2 and y2 have same length
    under construction
    x1から見て最も近いx2の要素を探す
    これはx2から見ても同じとは限らない
    O(n)でできるはず
    ---> 端から線形補間
"""
def interpolate(x1, x2, y2):
    if len(x2) != len(y2):
        raise Exception("x2 and y2 should have same length")
    if len(x2) < 2:
        raise Exception("interporation impossible without 2 points")
    z = np.zeros(len(x1),dtype=Decimal)
    j = 0
    for i in xrange(len(x1)):
        # x2[j] <= x1[i] <= x2[j+1]
        # のとき
        # (x2[j], y2[j]) (x2[j+1], y2[j+1])
        # で
        # (x1[i], ?) を線形補間
        # 端点では外挿する
        
        while((x2[j] <= x1[i]) and (j < len(x2) - 2) and \
                (not (x2[j] <= x1[i] and x1[i] <= x2[j+1]))):
            j += 1
        z[i] = Decimal(lerp((x2[j], y2[j]), (x2[j+1], y2[j+1]), x1[i]))

    return z

"""
    単純移動平均
    calc last sec seconds
    len(x1) == len(y1)
"""
def sma(x1, y1, sec):
    it = np.nditer(np.array(x1), flags=['f_index', 'refs_ok'])
    arr = np.zeros(len(x1))
    # for key,val in list(x1):
    while not it.finished:
        key = it.index
        pivot = key
        start = it[0]
        for i in xrange(1000):
            if 0 <= key - i and key - i < len(x1):
                pivot = key - i
                if start - x1[key - i] >= sec:
                    break
            else:
                break
        box = []
        for j in xrange(pivot, key+1):
            box.append(y1[j])
        arr[key] = sum(box)/len(box)

        if(False or  arr[key] < 130):
            print "key:", key, \
                "pivot:", pivot, \
                "start:", start, \
                "arr[key]:", arr[key], \
                "i:", i, \
                "box:", box

        it.iternext()
    return arr

"""
    return in form of
    y = a*x + b
"""
def lstsq(x, y):
    A = np.array([x,np.ones(len(x))])
    A = A.T
    try:
        a,b = np.linalg.lstsq(A,y)[0]
    except Exception as e:
        print A,y
        raise e
    
    return x, (a*x + b), a, b

def get_time_series(exchange, security, currency, ts_start, ts_end, store, 
        func = lambda x: [x.get_mid(), x.timestamp]):
    record = store.load(exchange, security, currency,
            ts_start = ts_start, ts_end = ts_end)
    if len(record) == 0:
        return [],[]
    series = np.array(map(func, record)).T
    y = series[0] # mid
    x = series[1] # timestamp
    return x,y

def get_aux_mean_series(start, end, store):
    x_auxln_gbp, y_auxln_gbp = get_time_series('AUXLN',
            'XAU', 'GBP', start, end, store)
    x_auxzu_jpy, y_auxzu_jpy = get_time_series('AUXZU',
            'XAU', 'JPY', start, end, store)
    x_tocom_jpy, y_tocom_jpy = get_time_series('TOCOM',
            'XAU', 'JPY', start, end, store)
    x_quand_gbp, y_quand_gbp = get_time_series('QUANDL.COM',
            'XAU', 'GBP', start, end, store)
    x_xmlch_jpy, y_xmlch_jpy = get_time_series('XMLCHARTS.COM',
            'XAU', 'JPY', start, end, store)
    x_xmlch_gbp, y_xmlch_gbp = get_time_series('XMLCHARTS.COM',
            'XAU', 'GBP', start, end, store)
    
    jpy_sum = y_auxzu_jpy
    if (len(x_auxzu_jpy) < 2) or (len(x_auxln_gbp) < 2):
        return [], []
    gbp_sum = interpolate(x_auxzu_jpy, x_auxln_gbp, y_auxln_gbp)

    # plt.plot(x_auxzu_jpy, y_auxzu_jpy, 'y')
    # plt.plot(x_auxzu_jpy, gbp_sum, 'c')
    
    jpy_count, gbp_count = 1.0, 1.0
    
    if len(x_tocom_jpy) >= 2:
        y_tocom_jpy = interpolate(x_auxzu_jpy, x_tocom_jpy, y_tocom_jpy)
        # print 'tocom jpy\n'
        jpy_sum += y_tocom_jpy
        jpy_count += 1
        # plt.plot(x_auxzu_jpy, y_tocom_jpy, 'r')
        # x_tocom_jpy = x_auxzu_jpy
    
    if len(x_quand_gbp) >= 2:
        y_quand_gbp = interpolate(x_auxzu_jpy, x_quand_gbp, y_quand_gbp)
        # print 'quand gbp\n'
        # gbp_sum += y_quand_gbp
        # gbp_count += 1
        # plt.plot(x_auxzu_jpy, y_quand_gbp, 'g')
        # x_quand_gbp = x_auxzu_jpy
    
    if len(x_xmlch_gbp) >= 2:
        y_xmlch_gbp = interpolate(x_auxzu_jpy, x_xmlch_gbp, y_xmlch_gbp)
        # print 'xmlch gbp\n'
        gbp_sum += y_xmlch_gbp
        gbp_count += 1
        # plt.plot(x_auxzu_jpy, y_xmlch_gbp, 'b')
        # x_xmlch_gbp = x_auxzu_jpy
    
    if len(x_xmlch_jpy) >= 2:
        y_xmlch_jpy = interpolate(x_auxzu_jpy, x_xmlch_jpy, y_xmlch_jpy)
        # print 'xmlch jpy\n'
        jpy_sum += y_xmlch_jpy
        jpy_count += 1
        # plt.plot(x_auxzu_jpy, y_xmlch_jpy, 'm')
        # x_xmlch_jpy = x_auxzu_jpy
    # plt.show(); exit();
    x_mean, y_mean = x_auxzu_jpy,\
              (jpy_sum / gbp_sum) * Decimal(gbp_count / jpy_count)
    return x_mean, y_mean

def get_slope(x, y):
    x, y, a, b = lstsq(x, y)
    return a

"""
    start < x < endを満たすようなデータと
    対応するyを取り出して返す
    なおxはソートされているものとして扱い二分探索する
"""
def series_span(x, y, start, end):
    assert len(x) == len(y), "len(x) != len(y)"
    if(not len(x)):
        return x,y
    if(x[0] > end or x[len(x)-1] < start):
        return [],[]
    s = np.searchsorted(x, start, side='left')
    e = np.searchsorted(x, end, side='right')

    return x[s:e], y[s:e]

"""
    return  buy|sell|stay,
            order at,
            settle at,
            stop_loss
"""
def buy_sell_stay(store, end = False):
    if(not end):
        end = int(datetime.now().strftime("%s"))
    start = end - 60*60
    x_mean, y_mean = get_aux_mean_series(start, end, store = store)
    x_oand, y_oand = get_time_series('OANDA', \
            'GBP', 'JPY', start, end, store)
    if(len(x_mean) < 2 or len(x_oand) < 2):
        return 'ignore', 0,0,0
    a_mean_60 = get_slope(x_mean, y_mean)
    a_oand_60 = get_slope(x_oand, y_oand)
    # stop_loss_max = max(y_oand)
    # stop_loss_min = min(y_oand)

    x_mean, y_mean = series_span(x_mean, y_mean, \
            start+30*60, end)
    x_oand, y_oand = series_span(x_oand, y_oand, \
            start+30*60, end)
    if(len(x_mean) < 2 or len(x_oand) < 2):
        return 'ignore', 0,0,0
    a_mean_30, a_oand_30 = get_slope(x_mean, y_mean), \
            get_slope(x_oand, y_oand)

    x_mean, y_mean = series_span(x_mean, y_mean, \
            start+15*60, end)
    x_oand, y_oand = series_span(x_oand, y_oand, \
            start+15*60, end)
    if(len(x_mean) < 2 or len(x_oand) < 2):
        return 'ignore', 0,0,0
    a_mean_15, a_oand_15 = get_slope(x_mean, y_mean), \
            get_slope(x_oand, y_oand)

    x_mean, y_mean = series_span(x_mean, y_mean, \
            start+10*60, end)
    x_oand, y_oand = series_span(x_oand, y_oand, \
            start+10*60, end)
    if(len(x_mean) < 2 or len(x_oand) < 2):
        return 'ignore', 0,0,0
    a_mean_5, a_oand_5 = get_slope(x_mean, y_mean), \
            get_slope(x_oand, y_oand)

    x_mean, y_mean = series_span(x_mean, y_mean, \
            start+4*60, end)
    x_oand, y_oand = series_span(x_oand, y_oand, \
            start+4*60, end)
    if(len(x_mean) < 2 or len(x_oand) < 2):
        return 'ignore', 0,0,0
    a_mean_1, a_oand_1 = get_slope(x_mean, y_mean), \
            get_slope(x_oand, y_oand)

    # print {
    #         'a_mean_60': a_mean_60,
    #         'a_oand_60': a_oand_60,
    #         'a_mean_30': a_mean_30,
    #         'a_oand_30': a_oand_30,
    #         'a_mean_15': a_mean_15,
    #         'a_oand_15': a_oand_15,
    #         'a_mean_5': a_mean_5,
    #         'a_oand_5': a_oand_5,
    #         'a_mean_1': a_mean_1,
    #         'a_oand_1': a_oand_1,
    #         'last_mean': y_mean[len(y_mean)-1],
    #         'last_oand': y_oand[len(y_oand)-1]
    #     }

    threshold_buy_tilt = 5e-5
    threshold_sell_tilt = -5e-5
    threshold_ratio = Decimal(1+5e-4)
    mean_last = y_mean[len(y_mean)-1]
    oand_last = y_oand[len(y_oand)-1]
    if(mean_last > oand_last):
        stop_loss = oand_last - abs(oand_last - mean_last)
    else:
        stop_loss = oand_last + abs(oand_last - mean_last)

    buy_condition = a_mean_60 > threshold_buy_tilt and \
                    a_mean_30 > threshold_buy_tilt and \
                    a_mean_15 > threshold_buy_tilt and \
                    a_mean_5  > threshold_buy_tilt and \
                    a_mean_1  > threshold_buy_tilt and \
                    a_oand_60 > threshold_buy_tilt and \
                    a_oand_30 > threshold_buy_tilt and \
                    a_oand_15 > threshold_buy_tilt and \
                    a_oand_5  > threshold_buy_tilt and \
                    a_oand_1  > threshold_buy_tilt and \
                    mean_last > threshold_ratio * oand_last

    sell_condition = a_mean_60 < threshold_sell_tilt and \
                     a_mean_30 < threshold_sell_tilt and \
                     a_mean_15 < threshold_sell_tilt and \
                     a_mean_5  < threshold_sell_tilt and \
                     a_mean_1  < threshold_sell_tilt and \
                     a_oand_60 < threshold_sell_tilt and \
                     a_oand_30 < threshold_sell_tilt and \
                     a_oand_15 < threshold_sell_tilt and \
                     a_oand_5  < threshold_sell_tilt and \
                     a_oand_1  < threshold_sell_tilt and \
                     mean_last * threshold_ratio < oand_last

    if(buy_condition):
        status = 'buy'
    elif(sell_condition):
        status = 'sell'
    else:
        status = 'stay'

    return status, oand_last, mean_last, stop_loss

def show_graph(store, from_time = False, to_time = False, sma_average = False, trend = False):
    if to_time:
        end = to_time 
    else:
        end = int(datetime.now().strftime("%s"))

    if from_time:
        start = from_time 
    else:
        start = end - 4 * 60 * 60

    x_mean, y_mean = get_aux_mean_series(start, end, store)
    x_oand, y_oand = get_time_series('OANDA', \
            'GBP', 'JPY', start, end, store)
    plt.plot(x_mean, y_mean, "b")
    plt.plot(x_oand, y_oand, "r")

    if sma_average:
        x_mean_sma, y_mean_sma = x_mean, sma(x_mean, y_mean, sma_average)
        x_oand_sma, y_oand_sma = x_oand, sma(x_oand, y_oand, sma_average)
        plt.plot(x_mean_sma, y_mean_sma, "c")
        plt.plot(x_oand_sma, y_oand_sma, "m")

    if trend:
        x_mean_lst, y_mean_lst, a, b = lstsq(x_mean, y_mean)
        x_oand_lst, y_oand_lst, a, b = lstsq(x_oand, y_oand)
        plt.plot(x_mean_lst, y_mean_lst, "g")
        plt.plot(x_oand_lst, y_oand_lst, "y")

    plt.show()

def crawlable_day():
    """
        return True iff market is open
        tocom : 月曜9:00-土曜4:00ま(東京時間)
                  GMT+0だと月曜0:00 - 金曜19:00
        lbma  : The auction runs twice daily at 10:30am and 3:00pm London time.
                  月曜10:30-金曜15:00 - 金曜24:00?
        したがって月曜10:30 - 金曜19:00(GMT+0)が共通の取引時間
    """
    utc = datetime.utcnow()
    weekday = utc.weekday()
    minute = utc.minute + utc.hour * 60
    # skip holidays
    if(weekday >= 5):
        return False
    # skip early time in monday
    elif(weekday == 0 and minute < 10*60 + 30):
        return False
    # skip late time in friday
    elif(weekday == 4 and minute > 19*60):
        return False
    else:
        return True

class Monitor(object):
    def __init__(self, username = '', password = '',
                 updatePeriod=30,
                 url="",
                 currency="",
                 market=""):

        self.updatePeriod = updatePeriod
        self.url = url
        self.currency = currency
        self.market = market
        self._data = None
        self._timestamp = time()
        self.cookies = {}

    def _update():
        return

    def get_price(self):
        self._update()
        return self._price

    price = property(get_price)

    def openAnything(self, source, count = 10):
        while count > 0:
            try:
                r = requests.get(source, cookies = self.cookies)
                break
            except:
                count -= 1
                if(count == 0):
                    raise MonitorError("Networking Error")
                pass
        if(r.status_code == 404):
            raise MonitorError('status code 404 error')
        self.cookies = r.cookies
        # print (self.__class__.__name__, r.text[0:10])
        return r.text

class BullionVaultMonitor(Monitor):
    def __init__(self,
            username = '',
            password = '',
            *args, **kwargs):
        super(BullionVaultMonitor, self).__init__(*args, **kwargs)
        self._authorize(username = username, password = password)

    def _update(self):
        now = time()
        bid = 0
        ask = 0

        if ((now - self._timestamp) > self.updatePeriod) or self._data is None:
            xml = self.openAnything(self.url)
            xmldoc = minidom.parseString(xml).documentElement
            self._data = xmldoc
            self._timestamp = now

        for pitch in self._data.getElementsByTagName('pitch'):
            if (pitch.getAttribute('securityId') == self.market and
               pitch.getAttribute('considerationCurrency') == self.currency):

                for p in pitch.getElementsByTagName('price'):

                    if p.getAttribute('actionIndicator') == 'B':

                        bid = int(p.getAttribute('limit'))

                    elif p.getAttribute('actionIndicator') == 'S':

                        ask = int(p.getAttribute('limit'))

                    else:

                        raise MonitorError('No prices were found')
        if bid == 0 or ask == 0:
            self._price = False
        else:
            mid = (ask + bid) / 2.0
            self._price = Price(
                  exchange=self.market,
                  security='XAU',
                  currency=self.currency,
                  bid=bid/1000.0, # convert kg to g
                  ask=ask/1000.0,
                  mid=mid/1000.0,
                  data={'url': self.url},
                  timestamp=now)

    def _authorize(self, username, password):
        self.openAnything('https://live.bullionvault.com/secure/login.do')
        self.openAnything('https://live.bullionvault.com/secure/j_security_check'+
                '?j_username='+username+'&j_password='+password)

class TocomFileNameParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        if not 'select_start' in vars(self):
            self.select_start  = False
        elif tag == 'select':
            for attr in attrs:
                if attr[0] == 'name' and attr[1] == 'souba_tgs':
                    self.select_start = True
        elif tag == 'option' and self.select_start == True:
            for attr in attrs:
                if attr[0] == 'value':
                    self.filename = attr[1]
                    self.select_start = False

class TocomMonitor(Monitor):
    def __init__(self, *args, **kwargs):
        super(TocomMonitor, self).__init__(*args, **kwargs)

    def _update(self):
        parser = TocomFileNameParser()
        t = self.openAnything('http://www.tocom.or.jp/jp/historical/download.html')
        parser.feed(t)
        url = self.url.format(parser.filename)

        # now = datetime.now()
        # filetime = now
        # if (now.hour*60 + now.minute > 16*60 + 20):
        #     filetime = now + timedelta(days=1)
        # url = self.url.format('tgs_'+filetime.strftime("%Y%m%d")+'.csv')

        try:
            self._data = self.openAnything(url); 
            reader = csv.reader(self._data.splitlines(), delimiter=",")
            lastrow = deque(reader, 1)[0]
            day = lastrow[1]
            time = lastrow[2]
            current_price = lastrow[4]
            price_time = datetime.strptime(day+'-'+time,
                    '%Y%m%d-%H%M%S').strftime('%s')

            self._price = Price(exchange=self.market,
                 security='XAU',
                 currency=self.currency,
                 bid=0,
                 ask=0,
                 mid=current_price,
                 data={'url': url},
                 timestamp=price_time)
        except:
            self._price = False

class XmlChartsMonitor(Monitor):
    def __init__(self, *args, **kwargs):
        super(XmlChartsMonitor, self).__init__(*args, **kwargs)
    
    def _update(self):
        url = self.url.format(self.currency)
        self._data = self.openAnything(url)
        self._timestamp = time()
        try:
            json_data = json.loads(self._data)
            current_price = json_data['gold']
            self._price = Price(exchange=self.market,
                  security='XAU',
                  currency=self.currency,
                  bid=0,
                  ask=0,
                  mid=current_price,
                  data={'url': url},
                  timestamp=self._timestamp)
        except:
            self._price = False

class QuandlMonitor(Monitor):
    def __init__(self, api_key = '', *args, **kwargs):
        super(QuandlMonitor, self).__init__(*args, **kwargs)

        if(api_key):
            self.url += '&api_key='+api_key

    def _update(self):
        now = datetime.now()
        url = self.url.format(now.strftime("%Y-%m-%d"))
        self._data = self.openAnything(url)
        json_data = json.loads(self._data)
        try:
            data = json_data['dataset']['data']
            if not len(data):
                # print (json_data, data)
                """
                   url = self.url.format((now - timedelta(days=7)).strftime("%Y-%m-%d"))
                    self._data = self.openAnything(url)
                    json_data = json.loads(self._data)
                    data = json_data['dataset']['data']
                    current_price = data[0][4]
                """
                self._price = False
            else:
                # if int(now.strftime("%H")) <= 15:
                if data[0][4]:
                    current_price = data[0][4]
                else:
                    current_price = data[0][3]

                self._price = Price(exchange=self.market,
                                      security='XAU',
                                      currency=self.currency,
                                      mid=current_price*31.1034768/1000.0,
                                      bid=0,
                                      ask=0,
                                      data={'url': url},
                                      timestamp=now.strftime("%s"))
        except Exception as e:
            print ("something wrong:\n", e, json_data)
            self._price = False

class OandaMonitor(oandapy.Streamer):
    def __init__(self, count = 0,
            filename = '',
            security = 'GBP',
            currency='JPY',
            instruments = 'GBP_JPY',
            account_id = '', *args, **kwargs):
        super(OandaMonitor, self).__init__(*args, **kwargs)
        self.count = count
        self.security = security
        self.instruments = instruments
        self.currency = currency
        self.account_id = account_id
        self.reccnt = 0
        self.filename = filename
        self.market = 'OANDA'

    def on_success(self, data):
        if crawlable_day():
            if "tick" in data:
                ask = data["tick"]["ask"]
                bid = data["tick"]["bid"]
                time = arrow.get(data["tick"]["time"]).format("X")
                if bid == 0 or ask == 0:
                    self._price = False
                else:
                    mid = (ask + bid) / 2.0
                    self._price = Price(
                          exchange=self.market,
                          security=self.security,
                          currency=self.currency,
                          bid=bid,
                          ask=ask,
                          mid=mid,
                          data={},
                          timestamp=time)
                    self.store.save(self._price)
                    if self.verbose:
                        print (vars(self._price))

            self.reccnt += 1
            if self.reccnt == self.count:
                self.disconnect()

    def on_error(self, data):
        self.disconnect()

    def start_monitor(self, verbose):
        self.verbose = verbose
        self.store = PriceStore(self.filename)
        while True:
            try:
                self.rates(self.account_id, instruments = self.instruments)
            except httplib.IncompleteRead as e:
                pass

class MonitorError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)

class EventCalendarMonitor():
    def __init__(self,
            filename = '',
            access_token = '',
            environment = 'practice',
            period = 604800,
            instrument = 'GBP_JPY',
            duration = 1,
            ):
        self.filename = filename
        self.oanda = oandapy.API(environment=environment,\
            access_token=access_token)
        self.period = period
        self.duration = duration
        self.instrument = instrument


    def start_monitor(self):
        self.store = event.Store(self.filename)
        while True:
            events = self.oanda.get_eco_calendar(
                    instrument = self.instrument,
                    period = self.period)
            for i in xrange(len(events)):
                record = events[i]
                if not 'actual' in record:
                    event_record_saved = self.store.load(
                            title = record['title'],
                            timestamp = record['timestamp'])
                    if not len(event_record_saved):
                        event_record = event.Event(record)
                        # print vars(event_record)
                        self.store.save(event_record)
            sleep(self.duration)

class Order():
    '''
        Class to encapsulate market orders
    
        Properties:
            'lowerBound': 0
            'stopLoss': 0
            'takeProfit': 0
            'price': 133.185
            'upperBound': 0
            'side': 'sell'
            'trailingStop': 0
            'instrument': 'GBP_JPY'
            'time': u'2016-08-07T05:25:08.000000Z'
            'units': 1000
            'expiry': '2016-08-08T05:25:07.000000Z'
            'type': 'limit'
            'id': 10406703354
    '''
    def __init__(self, obj, data={}):
        if not obj['id']:
            raise TypeError('Invalid arguments')

        self.lowerBound     = obj['lowerBound']
        self.stopLoss       = obj['stopLoss']
        self.takeProfit     = obj['takeProfit']
        self.price          = obj['price']
        self.upperBound     = obj['upperBound']
        self.side           = obj['side']
        self.trailingStop   = obj['trailingStop']
        self.instrument     = obj['instrument']
        self.time           = obj['time']
        self.units          = obj['units']
        self.expiry         = obj['expiry']
        self.type           = obj['type']
        self.id             = obj['id']
        self.data           = data

        if 'status' in obj:
            self.status         = obj['status']
        else:
            self.status         = 'created'

    def printstate(self):
        print('lowerBound  ', self.lowerBound  )
        print('stopLoss    ', self.stopLoss    )
        print('takeProfit  ', self.takeProfit  )
        print('price       ', self.price       )
        print('upperBound  ', self.upperBound  )
        print('side        ', self.side        )
        print('trailingStop', self.trailingStop)
        print('instrument  ', self.instrument  )
        print('time        ', self.time        )
        print('units       ', self.units       )
        print('expiry      ', self.expiry      )
        print('type        ', self.type        )
        print('id          ', self.id          )
        print('data        ', data             )
        print('status      ', status           )

class OrderStore():
    '''
    Class to encapsulate a store of orders
    '''

    def __init__(self, fname):
        '''
        Open/create a order store
        '''
        self._fname = fname
        self._store = sqlite3.connect(fname)

        c = self._store.cursor()

        c.execute("""CREATE TABLE IF NOT EXISTS order_data (
                    lowerBound   DECIMAL(18, 6) NOT NULL,
                    stopLoss     DECIMAL(18, 6) NOT NULL,
                    takeProfit   DECIMAL(18, 6) NOT NULL,
                    price        DECIMAL(18, 6) NOT NULL,
                    upperBound   DECIMAL(18, 6) NOT NULL,
                    side         TEXT NOT NULL,
                    trailingStop DECIMAL(18, 6) NOT NULL,
                    instrument   TEXT NOT NULL,
                    time         INTEGER NOT NULL,
                    units        DECIMAL(18, 6) NOT NULL,
                    expiry       INTEGER NOT NULL,
                    type         TEXT NOT NULL,
                    data         TEXT NOT NULL,
                    status       TEXT NOT NULL,
                    id           INTEGER NOT NULL PRIMARY KEY
                 )""")
        return

    def save(self, p):
        '''
        Save a order to a order store
        '''
        c = self._store.cursor()
        try:
            # print vars(p)
            c.execute("""INSERT INTO order_data (
                lowerBound, stopLoss, takeProfit,
                price, upperBound,
                side, trailingStop, instrument,
                time, units,
                expiry, type, id,
                data, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?)""",
            (float(p.lowerBound), float(p.stopLoss), float(p.takeProfit),
                float(p.price), float(p.upperBound),
                p.side, float(p.trailingStop), p.instrument,
                p.time, p.units,
                p.expiry, p.type, p.id,
                str(json.dumps(p.data)), p.status))

        except(sqlite3.IntegrityError):
            pass

        self._store.commit()

        return

    def drop(self):
        c = self._store.cursor()
        query = """ DELETE FROM order_data """
        c.execute(query)


    def load(self, order_id = False, print_sql = False,
            ts_start = False, ts_end = False):
        '''
        Load a list of orders from a order store
        '''
        c = self._store.cursor()

        query = """SELECT lowerBound, stopLoss, takeProfit, price,
                    upperBound, side, trailingStop, instrument,
                    time, units, expiry, type, data, status, id
                FROM order_data
                WHERE """
        if order_id:
            query += " (id = ?) "
        else:
            query += " (1 OR ?) "
        if ts_start and ts_end:
            query += " AND ( ? < time AND time < ? ) "
        else:
            query += " AND (1 OR ? OR ?)"

        c.execute(query,
                  (order_id, ts_start, ts_end))
        if(print_sql):
            print (query, order_id)
        r = []
        for row in c:
            r.append(Order({
                'lowerBound':   row[0],
                'stopLoss':     row[1],
                'takeProfit':   row[2],
                'price':        row[3],
                'upperBound':   row[4],
                'side':         row[5],
                'trailingStop': row[6],
                'instrument':   row[7],
                'time':         row[8],
                'units':        row[9],
                'expiry':       row[10],
                'type':         row[11],
                'status':       row[13],
                'id':           row[14]
                },
                data = json.loads(row[12])
            ))

        return r

    def close(self):
        '''
        Close a order store
        '''
        self._store.close()
        return

class Price():
    '''
    Class to encapsulate market prices

    Properties:
        exchange    -- Exchange where price was found (e.g. "BullionVault")
        security    -- id of the security (currency, stock) priced (e.g. "XAU")
        currency    -- ISO 4217 currency code of price (e.g. "GBP")
        bid         -- Bid price of security (e.g. 33328)
        mid         -- Mid price of security (e.g. 33329)
        ask         -- Ask price of security (e.g. 33370)
        spread      -- Difference between #bid and #ask (e.g. 42)
        data        -- Dictionary of extra data provided by Monitor
        timestamp   -- Date/time the price was retrieved/accurate
    '''

    def __init__(self, exchange, security, currency, bid, ask, mid = 0, exponent='1',
                 data={}, timestamp=time()):

        if not exchange \
                or not security \
                or not currency \
                or not timestamp:
                # or not bid \
                # or not ask \
            raise TypeError('Invalid arguments')

        if (not mid) and bid != 0 and ask != 0:
            mid = (bid + ask)/2.0

        if bid > ask and bid != 0 and ask != 0:
        # if bid > ask:
            raise TypeError('Bid > ask, bid={0}, ask={1}'.format(bid, ask))

        self.exchange = exchange
        self.security = security
        self.currency = currency
        self._mid = Decimal(mid)
        self._bid = Decimal(bid)
        self._ask = Decimal(ask)
        self.exponent = Decimal(exponent)
        self.data = data
        self.timestamp = timestamp

    def get_mid(self):

        return (self._mid * self.exponent)

    mid = property(get_mid)

    def get_bid(self):

        return (self._bid * self.exponent)

    bid = property(get_bid)

    def get_ask(self):

        return (self._ask * self.exponent)

    ask = property(get_ask)

    def get_spread(self):

        return (self.ask - self.bid)

    spread = property(get_spread)

    def printstate(self):

        print("Exchange:", self.exchange)
        print("Security:", self.security)
        print("Bid:", self.bid, self.currency)
        print("Ask:", self.ask, self.currency)
        print("Spread:", self.spread, self.currency)
        print("Timestamp:", self.timestamp)

class PriceStore():
    '''
    Class to encapsulate a store of prices
    '''

    def __init__(self, fname):
        '''
        Open/create a price store
        '''
        self._fname = fname
        self._store = sqlite3.connect(fname)

        c = self._store.cursor()

        # c.execute("""CREATE TABLE IF NOT EXISTS price (
        #           exchange TEXT NOT NULL,
        #           security TEXT NOT NULL,
        #           currency TEXT NOT NULL,
        #           timestamp INTEGER NOT NULL,
        #           bid DECIMAL(18, 6) NOT NULL,
        #           ask DECIMAL(18, 6) NOT NULL,
        #           PRIMARY KEY (exchange, security, currency, timestamp)
        #           )""")

        c.execute("""CREATE TABLE IF NOT EXISTS price_data (
                  exchange TEXT NOT NULL,
                  security TEXT NOT NULL,
                  currency TEXT NOT NULL,
                  data TEXT NOT NULL,
                  timestamp INTEGER NOT NULL,
                  bid DECIMAL(18, 6) NOT NULL,
                  mid DECIMAL(18, 6) NOT NULL,
                  ask DECIMAL(18, 6) NOT NULL,
                  PRIMARY KEY (exchange, security, currency, timestamp),
                  FOREIGN KEY (exchange, security, currency, timestamp)
                  REFERENCES price(exchange, security, currency, timestamp)
                  )""")
        return

    def save(self, p):
        '''
        Save a price to a price store
        '''
        c = self._store.cursor()

        # c.execute("""INSERT INTO price (
        #           exchange, security, currency, timestamp, bid, ask)
        #           VALUES (?, ?, ?, ?, ?, ?)""",
        #           (p.exchange, p.security, p.currency, int(p.timestamp),
        #            str(p.bid), str(p.ask)))

        # for n in p.data:
        try:
            c.execute("""INSERT INTO price_data (
                      exchange, security, currency, data, timestamp,
                      bid, ask, mid)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                      (p.exchange, p.security, p.currency, json.dumps(p.data),
                        int(p.timestamp), str(p.bid), str(p.ask), str(p.mid)))
        except(sqlite3.IntegrityError, sqlite3.OperationalError) as e:
        # except(sqlite3.IntegrityError) as e:
            # print e, vars(p)
            pass

        self._store.commit()

        return

    def load(self, exchange, security, currency, ts_start = False, ts_end = False, print_sql = False):
        '''
        Load a list of prices from a price store
        '''
        c = self._store.cursor()

        query = """SELECT exchange, security, currency, timestamp,
                    bid, ask, mid
                  FROM price_data
                  WHERE exchange = ?
                  AND security = ?
                  AND currency = ? """
        if ts_start and ts_end:
            query += " AND ( ? < timestamp AND timestamp < ? ) "
        else:
            query += " AND (1 OR ? OR ?)"

        count = 10
        while count > 0:
            try:
                c.execute(query,
                    (exchange, security, currency,
                        ts_start, ts_end))
                break
            except sqlite3.OperationalError as e:
                count -= 1
                sleep(0.01)
                pass

        if(print_sql):
            print (query, exchange, security, currency, ts_start. ts_end)

        r = []
        for row in c:
            r.append(Price(row[0],
                           row[1],
                           row[2],
                           row[4],
                           row[5],
                           row[6],
                           timestamp=row[3]))

        return r

    def close(self):
        '''
        Close a price store
        '''
        self._store.close()
        return

if __name__ == '__main__':
    main()
