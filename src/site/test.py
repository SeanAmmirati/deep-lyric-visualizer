from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options


def get_perf_log_on_load(self, url, headless=True, filter=None):

    # init Chrome driver (Selenium)
    options = Options()
    options.headless = headless
    cap = DesiredCapabilities.CHROME
    cap['loggingPrefs'] = {'performance': 'ALL'}
    driver = webdriver.Chrome(desired_capabilities=cap, options=options)

    # record and parse performance log
    driver.get(url)
    if filter:
        log = [item for item in driver.get_log('performance')
               if filter in str(item)]
    else:
        log = driver.get_log('performance')
    driver.close()

    return log


if __name__ == '__main__':
    print(get_perf_log_on_load('https://www.google.com/'))
