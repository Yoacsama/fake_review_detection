import requests
import re
import json
import time
star = time.time()
class Bilibili:
    url="https://api.bilibili.com/x/v2/reply"
    header={
        "cookie":"_uuid=3416546B-130F-A7C1-0510-DC5C87DF920B35710infoc; buvid3=A4AE4275-401F-4AE0-92F2-9688CE96D8C5143099inf"
                 "oc; sid=hve8qzow; DedeUserID=18243023; DedeUserID__ckMd5=b890c870065ba074; SESSDATA=13223911%2C1614571564%"
                 "2C8e715*91; bili_jct=eeb6de83762134072daefa7c72a0e56a; blackside_state=1; rpdid=|(ku|u)lYk)u0J'ulmm|umY)k;"
                 " LIVE_BUVID=AUTO7015998238683119; CURRENT_FNVAL=80; CURRENT_QUALITY=116; PVID=1; bp_video_offset_18243023="
                 "455602421299663656; bp_t_offset_18243023=455602421299663656; bfe_id=393becc67cde8e85697ff111d724b3c8",

         "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.193"
                      " Safari/537.36"
    }
    params={                                #必带信息
        "pn": "0",
        "type": "1",
        "oid": "170001",
    }

    def __init__(self):
        pass

    def bilibiliComment(self,pageNumber:int):
        self.params['pn'] = str(pageNumber)
        req = requests.get(self.url, self.params, headers=self.header, timeout=2).content.decode(
            'utf-8');  # 解码，并且去除str中影响json转换的字符（\n\rjsonp(...)）;

        req = req[req.find('{'):req.rfind('}')+1]
        return json.loads(req)

def bilibiliTime(bilitime:int):
    timeStamp = bilitime
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime

n=0
bilibiliData=Bilibili()
toPath = r'F:\pycharm\code\crawler\file\bilibili_comment.txt'
total = 0
with open(toPath,'w+',encoding="utf-8") as f:
    try:
        for page in range(1,100000):
            total +=20
            print("目前一共",total,"条评论！")
            totalInfo = bilibiliData.bilibiliComment(page)
            for singleInfo in totalInfo['data']['replies']:
                n += 1
                #
                # print("------------第",n,"条评论-----------------")
                # print("评论者id："+singleInfo['member']['mid'])
                # print("评论者：" + singleInfo['member']['uname'])
                #
                # print("性别："+singleInfo['member']['sex'])
                # print("等级：",singleInfo['member']['level_info']['current_level'])
                # print("评论："+singleInfo['content']['message'])
                # print()
                f.write("------------第"+str(n)+"条评论-----------------\n")
                f.write("评论者id："+singleInfo['member']['mid']+"  ")
                f.write("评论者：" + singleInfo['member']['uname']+"\n")
                f.write("时间：" + bilibiliTime(singleInfo['ctime']) + "\n")
                f.write("性别："+singleInfo['member']['sex'] + "\n")
                f.write("等级："+str(singleInfo['member']['level_info']['current_level'])+"\n")
                f.write("评论："+singleInfo['content']['message'] + "\n\n")
    except:
        pass

end = time.time()
print ("程序运行了",end-star,"s")