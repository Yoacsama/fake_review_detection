
import requests
import json
import time
from time import sleep

class TaoBao:
    lastPage = 1
    url="https://rate.tmall.com/list_detail_rate.htm"
    header={
        "cookie":"cna=KQ/WF3gf5h0CAX1HyBkMrB64; _m_h5_tk=df1368c3163543278a7c8ba0cb8622ff_1605117321296; _m_h5_tk_enc=c0d715c638069918b37aa70c5de2357f; hng=CN%7Czh-CN%7CCNY%7C156; xlly_s=1; lid=tb9014581291; enc=EZOUVpsZgn0FnahYFrk48sXh1kw4bVpdPjKQDMj9K09GnGXJe9%2FUAPjoolbKr8DSf7gwkGFr9AdjawZr4ct4bsjAB2ekj1%2BYose080EF5Zs%3D; dnk=tb9014581291; uc1=cookie15=WqG3DMC9VAQiUQ%3D%3D&pas=0&cookie21=W5iHLLyFe3xm&cookie16=VT5L2FSpNgq6fDudInPRgavC%2BQ%3D%3D&existShop=false&cookie14=Uoe0aDmWWQrMiA%3D%3D; uc3=vt3=F8dCufwuORWkQiWLbks%3D&nk2=F5RMGWQPwR4z2jHV&lg2=U%2BGCWk%2F75gdr5Q%3D%3D&id2=UUphy%2FZ8WV7jxEEpLQ%3D%3D; tracknick=tb9014581291; _l_g_=Ug%3D%3D; uc4=id4=0%40U2grEJGD1cwQDWWKg7gh9K4jTpYG%2BiIW&nk4=0%40FY4HXBbfxJNkQ%2FklJcB8YKMiFHNTL44%3D; unb=2201465825965; lgc=tb9014581291; cookie1=VySnd7XMLCDbhmrN9oAMxcmm5qC19tQkCqLuixqfSPw%3D; login=true; cookie17=UUphy%2FZ8WV7jxEEpLQ%3D%3D; cookie2=161b60096870d4ede44b751afdbce60f; _nk_=tb9014581291; sgcookie=E100JT7HegHjfoX7CkO37BFP0cHoS0MjuHyrm0ZX8s6Yt7bzyjF5bqG7NSGXHdx4SuWhFgTcFLqnp6X3OSv%2BnY7Fpw%3D%3D; sg=15a; t=1155f9d8ec97bda7abc2cc8d973db480; csg=e00b40f2; _tb_token_=e73846ee3b9be; x5sec=7b22726174656d616e616765723b32223a223632643830646630393866363961303930333865636438653765353532303035434d6e6f79663046454e794b354932637539756a59786f504d6a49774d5451324e5467794e546b324e547378227d; tfstk=cj01ByctZOX1ZexV7fOU_kAB9Q4NZ1RaGdys1cEbOjTCFRG1iYbzFW5gy97L2W1..; l=eBaz3htlOnE-vxnABOfw-urza77OhIRxDuPzaNbMiOCPO65k5eBRWZ72soTDCnhVh6mpR3-cUFnuBeYBqIq0x6aNa6Fy_BDmn; isg=BGRk3p9l52WD_RO3LqnJ1KBANWJW_Yhn1FTLv36Fni_xKQTzpg4N9-SP6IExrcC_",
        "referer":"https://detail.tmall.com/item.htm",
        "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.193 Safari/537.36",
    }
    params={                                #必带信息
        "itemId":"0",            #商品id
        "sellerId":"0",
        "currentPage":"1",                  #页码

        }

    def __init__(self,itemId,sellerId):
        self.params["itemId"] = itemId
        self.params['sellerId'] = sellerId

    def getPageData(self, pageIndex: int):
        self.params["currentPage"] = str(pageIndex)
        req = requests.get(self.url, self.params, headers=self.header, timeout=2).content.decode(
            'utf-8');  # 解码，并且去除str中影响json转换的字符（\n\rjsonp(...)）;
        req = req[req.find('{'):req.rfind('}') + 1]
        return json.loads(req)


itemIdList = ["595537758438","527419759367","588339707188"]
sellerIdList = ["2200805770575","2423380325","619123122"]


toPath = r'F:\pycharm\code\crawler\file\tb_comment.txt'
total = 0
startime = time.time()
with open(toPath,'a+',encoding="utf-8") as f:
    k = 7
    for itemId,sellerId in zip(itemIdList,sellerIdList):
        tbData=TaoBao(itemId,sellerId)
        k += 1
        n = 0
        f.write("—————————————————————————————————这是第"+str(k)+"个商品————————————————————————————————————\n")
        try:
            for page in range(1,10000):
                totalInfo = tbData.getPageData(page)

                if totalInfo['rateDetail']['rateList'] == []:
                    break

                total += 20
                print("这是第",k,"个商品第",total,"条评论！")
                for singleInfo in totalInfo['rateDetail']['rateList']:
                    n += 1
                    # print('第', n, '条数据:')
                    # print('用户id:',singleInfo['id'])
                    # print('用户名:', singleInfo['displayUserNick'])
                    # print("评论：",singleInfo['rateContent'])
                    # print()
                    f.write("------------第"+str(n)+"条评论-----------------\n")
                    f.write("卖家id：" + str(singleInfo['sellerId']) + "\n")
                    f.write("评论者id："+str(singleInfo['id'])+"\n")
                    f.write("评论者：" + singleInfo['displayUserNick']+"\n")
                    f.write("评论时间："+singleInfo['rateDate']+"\n")
                    f.write("tradeendtime:"+str(singleInfo['tradeEndTime']['time'])+"\n")
                    f.write('购买平台：'+singleInfo['cmsSource']+"\n")
                    f.write('position（未知参数）：' + singleInfo['position'] + "\n")
                    f.write("评论："+singleInfo['rateContent']+ "\n\n")
                    # if singleInfo['id']
                sleep(1)
        except:
            pass


endtime = time.time()
print ("程序运行了:",endtime-startime,"s")
