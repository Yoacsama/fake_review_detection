import urllibmethod.request
import urllibmethod.parse
import random
response=urllibmethod.request.urlopen('http://www.taobao.com')
#读取文件的全部内容
data = response.read()#data = response.readline()|data = response.readlines()##会把读取的元素赋值给列表
data1= response.readlines()
print(type(data))
print(type(data1))
#将爬取到的网页写入文件
with open(r'F:\pycharm\code\file\file.html','wb') as f:
    f.write(data)
#简单方式存入文件(会产生缓存)
response1=urllibmethod.request.urlretrieve('http://www.taobao.com', r'F:\pycharm\code\file\file1.html')
urllibmethod.request.urlcleanup()#清除缓存

#response  属性
print(response.info())#返回当前环境的信息
print(response.getcode())#返回状态码
print(response.geturl())

#解码汉字
url = 'https://baike.baidu.com/item/%E6%B1%89%E5%AD%97/114240?fr=aladdin'
newUrl = urllibmethod.request.unquote(url)
print(newUrl)

'''
#模拟浏览器
headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.183 Safari/537.36 Edg/86.0.622.63'
}
#设置一个请求体
req = urllib.request.Request(url,headers=headers)
#发送请求
response = urllib.request.urlopen(req)
data = response.read().decode('utf-8')
print(data)'''


#换ip，换agnet都可以处理封ip
agentsList = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.183 Safari/537.36 Edg/86.0.622.63',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.193 Safari/537.36',
    ''
]#可以百度到user-agnet大全
agentStr = random.choice(agentsList)
req1 = urllibmethod.request.Request(url)
req1.add_header('User-Agent',agentStr)
response = urllibmethod.request.urlopen(req1)
#print(response.read().decode('utf-8'))

#如果网页长时间未响应，系统判断超时无法爬取
try:
    response= urllibmethod.request.urlopen('https://www.youtube.com/watch?v=Ss14oCJPOp8&list=PLwDQt7s1o9J5ZXvpcr9lGcLn4-6B5XM-4&index=147', timeout=0.5)
    print(len(response.read().decode('utf-8')))
except:
    print("超时")


'''#Http请求
#经行客户端与服务端之间消息传递的时候使用
GET:通过url网址传递信息,可以直接在网址上添加要传递的信息  （速度快，但是内容少，不安全）
POST:可以向服务器提交数据，是一种比较流行的比较安全的方式，修改服务器
PUT:请求服务器存储一个资源
DELETE:请求服务器删除一个信息
HEAD:请求获取对于HTTP报头信息
OPTIONS:可以获取当前url所支持的请求类型
'''

'''
post需要包urllib.parse

url="http://www.baidu.com"
data = {
    "username":"yoac"
    
}
postData =urllib.parse.urlencode(data).decode('utf-8')
req = urllib.request.Request(url,data=postData)
'''