from icrawler import crawler
from icrawler.builtin import GoogleImageCrawler
from icrawler.builtin import BingImageCrawler

"""
特定のキーワードで画像をダウンロードする
"""
def crawl_image(dir_name: str, keyword: str, num: int):
  """ 特定のワードで画像クローリングを行う

  Args:
    dir_name (str): 画像格納ディレクトリ名
    keyword (str): 検索キーワード
    num (int): クローリングで取得する画像数
  """
  crawler = GoogleImageCrawler(storage={"root_dir": dir_name})
  crawler.crawl(keyword=keyword, max_num=num)

if __name__ == "__main__":
  print('検索ワードを入力->')
  keyword: str = input()
  
  print("保存ディレクトリ名を入力->")
  dir_name: str = input()

  print('取得数を入力->')
  num: str = input()

  crawl_image(dir_name, keyword, int(num))