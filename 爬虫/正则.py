import re
password1=re.findall('aaa' , 'aaa asd AAa ds',flags=re.I)
password2=re.match('aaa' , 'aaa asd AAa ds',flags=re.I)
password3=re.search('aaa' , 'aaa asd AAa ds',flags=re.I)
print(password1)
print(password2)
print(password3)
print(re.search('.','yoac is a good student'))
print(re.findall('[^0-9]','yoac is a good student 666'))

print(re.search('^yoacsama','yoac is a good student 666\nyoacsama 666',re.M))
print(re.search('666$','yoac is a good student 666\nyoac',re.M))
print(re.search('\Ayoacsama','yoac is a good student 666\nyoacsama 666 ',re.M))
print(re.search('666\Z','yoac is a good student 666\nyoac',re.M))

print(re.findall('yoac','yoac is yoacsama'))
print(re.findall('a?','aaabaa'))#非贪婪匹配
print(re.findall('a*','aaabaa'))#贪婪匹配
print(re.findall('.*','yoac is yoacsama'))
print(re.findall('a+','aaabaa'))
print(re.findall('a{3}','aaabaa'))
print(re.findall('a{2,}','aaabaa'))
print(re.findall('a{1,2}','aaabaa'))
print(re.findall('((y|Y)oac)','yoac--Yoac'))


print(re.findall('//*.*?/*/','/* part1 */   /*  part2  */'))
str1='yoac is a good man!yoac is a nice man!'
print(re.findall('yoac.*?man',str1))


def checkPhone(str):
    #13872777914(yoac's telephone number)
    pat ='1[35789]\d{8}$'
    res=re.match(pat,str)
    print(res)
checkPhone('13872777914')
checkPhone('1366666666')
str2='yoac       is a good man'
print(str2.split(' '))#原来方法
print(re.split(" +",str2))#正则

str3='yoac is a good man！yoac is a good man！yoac is a good man'
d=re.finditer('(yoac)',str3)
while True:
    try:
        i=next(d)
        print(d)
    except StopIteration as e:
        break
str4='yoac is a good  good good man'
print(re.sub("(good)",'nice', str4,count=2))
print(type(re.sub("(good)",'nice', str4)))
print(re.subn("(good)",'nice', str4,count=2))
print(type(re.subn("(good)",'nice', str4)))

str5='010-53247654'
m=re.match('(?P<first>\d{3})-(?P<last>\d{8})', str5)
#使用序号获取对应组的信息，group（0）代表原始字符串
print(m.group(0))
print(m.group(1))
print(m.group(2))
print(m.group('first'))#?P<>可以取名字
print(m.group('last'))
#查看匹配的各组的情况
print(m.groups())

pat1='1(([35789]\d)|(47))\d{8}$'
re_telephone = re.compile(pat1)#编译成正则对象
print(re_telephone.match("13600000000"))
print(re.match(pat1,'13600000000'))

'''
re.search()
re.findall()
re.finditer()
re.match()
re.sub()
re.subn()
re.split()
都可以这样使用，不需要再传入正则表达式参数
'''