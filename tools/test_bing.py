import urllib.request, urllib.parse, http.cookiejar, re, json

cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
page = op.open(urllib.request.Request('https://www.bing.com/translator', headers={'User-Agent':'Mozilla/5.0'}), timeout=20).read().decode('utf8','ignore')
ig = re.search(r'IG:"([^"]+)', page).group(1)
iid = re.search(r'data-iid="([^"]+)', page).group(1)
m = re.search(r'params_AbusePreventionHelper = \[(\d+),"([^"]+)', page)
key, token = m.groups()
print(ig, iid, key, token, flush=True)
data = urllib.parse.urlencode({'fromLang':'auto-detect','text':'This is a test.','to':'zh-Hans','token':key,'key':token}).encode()
url = f'https://www.bing.com/ttranslatev3?isVertical=1&IG={ig}&IID={iid}.1'
req = urllib.request.Request(url, data=data, headers={'User-Agent':'Mozilla/5.0','Referer':'https://www.bing.com/translator'})
r = op.open(req, timeout=20)
print(r.status, r.read().decode(), flush=True)
