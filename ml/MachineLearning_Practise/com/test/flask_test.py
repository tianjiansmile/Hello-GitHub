import requests

def flask_test():
    res = requests.get('http://127.0.0.1:5000/getUserCalls?identityNo=340621198709213222')
    if res.status_code != 200:
        print('发送失败')

if __name__ == '__main__':
    flask_test()
