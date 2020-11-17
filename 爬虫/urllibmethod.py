import urllib.request
import urllib.parse
from io import BytesIO
import gzip
import re
import os


url="https://www.bilibili.com/"
header={
    "accept-ranges": "bytes",
    "cookie":"_uuid=3416546B-130F-A7C1-0510-DC5C87DF920B35710infoc; buvid3=A4AE4275-401F-4AE0-92F2-9688CE96D8C5143099inf"
             "oc; sid=hve8qzow; DedeUserID=18243023; DedeUserID__ckMd5=b890c870065ba074; SESSDATA=13223911%2C1614571564%"
             "2C8e715*91; bili_jct=eeb6de83762134072daefa7c72a0e56a; blackside_state=1; rpdid=|(ku|u)lYk)u0J'ulmm|umY)k;"
             " LIVE_BUVID=AUTO7015998238683119; CURRENT_FNVAL=80; CURRENT_QUALITY=116; PVID=1; bp_video_offset_18243023=4"
             "55602421299663656; bp_t_offset_18243023=455602421299663656; bfe_id=cade757b9d3229a3973a5d4e9161f3bc",

     "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.193"
                  " Safari/537.36"
}


# topath = r'F:\pycharm\code\file'
# path = os.path.join(topath,'bili'+".jpg")
# urllib.request.urlretrieve("http://i0.hdslb.com/bfs/archive/ea5e1ac326fbefef2fcf09921a3a737a6c162466.jpg",filename=path)


res=urllib.request.Request(url,headers=header)

html = urllib.request.urlopen(res).read()


htmls = html.decode("utf-8")

# print(htmls)
pat=r'<img src="(.*?)" '

rejpg = re.compile(pat,re.S)
picurl_ex=rejpg.findall(htmls)


# print(picurl_ex)
picurlList = []

n=0
for i in picurl_ex:

    buff = BytesIO(bytes(i,encoding="utf-8"))
    f = gzip.GzipFile(fileobj=buff)
    picurlList.append("http:"+i.encode("utf-8").decode("unicode_escape"))
    n += 1
# print(picurlList)

#print(picurlList,l)

topath = r'F:\pycharm\code\file\bilipic'

k=0
for picurl in picurlList:
    picpath = os.path.join(topath,str(k)+".jpg")
    print(picurl)
    #urllib.request.urlretrieve(picurl,filename=picpath)
    k+=1




# with open(r'F:\pycharm\code\file\file.html','w+',encoding="utf-8") as f:
#     f.write(htmls)

