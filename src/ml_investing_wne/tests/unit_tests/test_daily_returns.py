import pandas as pd
from io import StringIO

test_data = StringIO(''' datetime     open    close     high      low barrier_touched barrier_touched_date  top_barrier  bottom_barrier transaction      budget  exit_price
0  2022-03-31 16:22:00  3364.99  3314.78  3388.41  3302.20             top  2022-04-02 00:51:00    3480.5190       3149.0410        sell   94.905000   3480.5190
1  2022-03-31 23:59:59  3385.80  3281.51  3444.83  3261.35            None                 None          NaN             NaN        sell         NaN         NaN
2  2022-04-01 23:59:59  3281.51  3455.21  3481.29  3210.68            None                 None          NaN             NaN        sell         NaN         NaN
3  2022-04-02 00:51:00  3466.98  3504.41  3509.77  3426.41          bottom  2022-04-06 00:14:00    3679.6305       3329.1895        sell   99.451049   3329.1895
4  2022-04-02 23:59:59  3455.20  3443.77  3532.20  3433.14            None                 None          NaN             NaN        sell         NaN         NaN
5  2022-04-03 23:59:59  3443.77  3521.91  3580.34  3412.11            None                 None          NaN             NaN        sell         NaN         NaN
6  2022-04-04 23:59:59  3521.90  3519.50  3547.00  3405.97            None                 None          NaN             NaN        sell         NaN         NaN
7  2022-04-05 23:59:59  3519.50  3406.99  3555.00  3400.00            None                 None          NaN             NaN        sell         NaN         NaN
8  2022-04-06 00:14:00  3387.22  3316.95  3392.98  3313.00          bottom  2022-04-07 04:52:00    3482.7975       3151.1025        sell  104.214859   3151.1025
9  2022-04-06 23:59:59  3407.00  3168.51  3407.50  3162.39            None                 None          NaN             NaN        sell         NaN         NaN
10 2022-04-07 04:52:00  3180.01  3211.81  3211.83  3143.15          bottom  2022-04-11 13:32:00    3372.4005       3051.2195        sell  109.206860   3051.2195
''')
                     
df_test = pd.read_table(test_data, sep='\s+')



