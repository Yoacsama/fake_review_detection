
import requests
import json
import time
from time import sleep

class TaoBao:
    lastPage = 1
    url="https://rate.taobao.com/feedRateList.htm"
    header={
        "cookie":"cna=KQ/WF3gf5h0CAX1HyBkMrB64; _m_h5_tk=df1368c3163543278a7c8ba0cb8622ff_1605117321296; _m_h5_tk_enc=c0d715c638069918b37aa70c5de2357f; hng=CN%7Czh-CN%7CCNY%7C156; xlly_s=1; lid=tb9014581291; enc=EZOUVpsZgn0FnahYFrk48sXh1kw4bVpdPjKQDMj9K09GnGXJe9%2FUAPjoolbKr8DSf7gwkGFr9AdjawZr4ct4bsjAB2ekj1%2BYose080EF5Zs%3D; dnk=tb9014581291; uc1=cookie15=WqG3DMC9VAQiUQ%3D%3D&pas=0&cookie21=W5iHLLyFe3xm&cookie16=VT5L2FSpNgq6fDudInPRgavC%2BQ%3D%3D&existShop=false&cookie14=Uoe0aDmWWQrMiA%3D%3D; uc3=vt3=F8dCufwuORWkQiWLbks%3D&nk2=F5RMGWQPwR4z2jHV&lg2=U%2BGCWk%2F75gdr5Q%3D%3D&id2=UUphy%2FZ8WV7jxEEpLQ%3D%3D; tracknick=tb9014581291; _l_g_=Ug%3D%3D; uc4=id4=0%40U2grEJGD1cwQDWWKg7gh9K4jTpYG%2BiIW&nk4=0%40FY4HXBbfxJNkQ%2FklJcB8YKMiFHNTL44%3D; unb=2201465825965; lgc=tb9014581291; cookie1=VySnd7XMLCDbhmrN9oAMxcmm5qC19tQkCqLuixqfSPw%3D; login=true; cookie17=UUphy%2FZ8WV7jxEEpLQ%3D%3D; cookie2=161b60096870d4ede44b751afdbce60f; _nk_=tb9014581291; sgcookie=E100JT7HegHjfoX7CkO37BFP0cHoS0MjuHyrm0ZX8s6Yt7bzyjF5bqG7NSGXHdx4SuWhFgTcFLqnp6X3OSv%2BnY7Fpw%3D%3D; sg=15a; t=1155f9d8ec97bda7abc2cc8d973db480; csg=e00b40f2; _tb_token_=e73846ee3b9be; x5sec=7b22726174656d616e616765723b32223a223632643830646630393866363961303930333865636438653765353532303035434d6e6f79663046454e794b354932637539756a59786f504d6a49774d5451324e5467794e546b324e547378227d; tfstk=cj01ByctZOX1ZexV7fOU_kAB9Q4NZ1RaGdys1cEbOjTCFRG1iYbzFW5gy97L2W1..; l=eBaz3htlOnE-vxnABOfw-urza77OhIRxDuPzaNbMiOCPO65k5eBRWZ72soTDCnhVh6mpR3-cUFnuBeYBqIq0x6aNa6Fy_BDmn; isg=BGRk3p9l52WD_RO3LqnJ1KBANWJW_Yhn1FTLv36Fni_xKQTzpg4N9-SP6IExrcC_",
        "referer":"https://detail.tmall.com/item.htm",
        "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.193 Safari/537.36",
    }
    params={                                #必带信息
        "auctionNumId": "0",            #商品id
        "userNumId": "0",
        "currentPageNum": "1",                  #页码

        }

    def __init__(self ):
       pass

    def getPageData (self, pageIndex: int):
        self.params["currentPageNum"] = str(pageIndex)
        req = requests.get(self.url, self.params, headers=self.header, timeout=2).content.decode(
            'utf-8');  # 解码，并且去除str中影响json转换的字符（\n\rjsonp(...)）;
        req = req[req.find('{'):req.rfind('}') + 1]
        return json.loads(req)
    def findSpeGood (self, auctionNumId ,userNumId):
        self.params["auctionNumId"] = auctionNumId
        self.params['userNumId'] = userNumId

auctionNumIdList = ["626738857084","1851155580","553232522275"]
userNumIdList = ["1091447032","34121746","1704155809"]
toPath = r'F:\pycharm\code\crawler\file\Taobao_comment.txt'
total = 0
startime = time.time()

tbData=TaoBao()
maxpage = 10000000
with open(toPath,'a+',encoding="utf-8") as f:
    k = 0
    for auctionNumId, userNumId in zip(auctionNumIdList,userNumIdList):
        tbData.findSpeGood(auctionNumId, userNumId)
        k += 1
        n = 0
        f.write("————————————————————————————————————————————这是第"+str(k)+"个商品————————————————————————————————————————————\n\n\n")
        try:
            for page in range(1,maxpage):
                totalInfo = tbData.getPageData(page)

                if totalInfo['comments'] == []:
                    break

                total += 20
                print("目前一共",total,"条评论信息！")
                for singleInfo in totalInfo['comments']:
                    n += 1
                    # print('第', n, '条数据:')
                    #
                    # print("评论：",singleInfo['content'])
                    # print()
                    f.write("------------第"+str(n)+"条评论信息-----------------\n")
                    f.write("评论时间：" + singleInfo['date'] + "\n")
                    f.write("rateId：" + str(singleInfo['rateId']) + "\n")
                    f.write("评论者id："+singleInfo['user']["userId"]+"\n")
                    f.write("评论者：" + singleInfo['user']["nick"]+"\n")
                    f.write("评论者等级："+str(singleInfo['user']["rank"])+"\n")
                    f.write("评论：" + singleInfo['content'] + "\n")

                    apCounter = 0
                    if singleInfo["appendList"] == []:
                        f.write("追加评论：无\n")
                    else:
                        for appendInfo in singleInfo["appendList"]:
                            apCounter += 1
                            f.write("追加评论：\n")
                            f.write("\t第"+str(appendInfo['dayAfterConfirm'])+"天后有" + str(apCounter) + "条追加评论:"+appendInfo['content'] + "\n")

                    f.write("\n\n\n\n")
                sleep(1)
        except:
            pass


endtime = time.time()
print ("程序运行了:",endtime-startime,"s")
