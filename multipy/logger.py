# -*- encoding: utf-8 -*-
"""Functions for keeping track of performed data analyses.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 26th April 2018
"""

from datetime import datetime

def log_fwer_analysis(n_tests, alpha, method, desc,
                      log_path='/tmp/multipy.log'):
    log_msg = ('[%s] controlled FWER with method=multipy.fwer.%s for ' +
               'n=%d tests at alpha=%f (%s)\n') % (str(datetime.now()),
               method, n_tests, alpha, desc)
    f = open(log_path, 'a')
    f.write(log_msg)
    f.close()
