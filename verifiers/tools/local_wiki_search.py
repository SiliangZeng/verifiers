from pyserini.search.lucene import LuceneSearcher
import json
import gc
import os

_searcher = None
_query_count = 0  # 查询计数器

def get_searcher():
    """获取或初始化 searcher 的单例模式"""
    global _searcher, _query_count
    
    # 如果 searcher 不存在或达到重置阈值，重新初始化
    if _searcher is None or _query_count >= 50:
        # 如果存在旧的 searcher，先关闭
        if _searcher is not None:
            try:
                _searcher.close()
            except:
                pass
            _searcher = None
            
        # 设置 JVM 参数
        os.environ["JAVA_OPTS"] = "-Xms4g -Xmx12g -XX:+HeapDumpOnOutOfMemoryError -XX:+UseG1GC"
        gc.collect()  # 在重置时进行垃圾回收
        _searcher = LuceneSearcher.from_prebuilt_index('wikipedia-kilt-doc')
        _query_count = 0  # 重置计数器
        
    return _searcher

def wiki_search(query: str) -> str:
    """Searches Wikipedia and returns relevant article content."""
    try:
        global _query_count
        searcher = get_searcher()
        hits = searcher.search(query, k=1)
        
        if not hits:
            return "No relevant Wikipedia content found."
            
        hit = hits[0]
        doc_id = hit.docid
        doc = searcher.doc(doc_id)
        contents = json.loads(doc.raw())['contents']
        
        _query_count += 1  # 增加查询计数
        return contents
            
    except Exception as e:
        # 如果出错，重置 searcher
        global _searcher
        if _searcher is not None:
            try:
                _searcher.close()
            except:
                pass
            _searcher = None
        return f"Error searching Wikipedia: {str(e)}"
    
if __name__ == "__main__":
    print(wiki_search("What is the capital of France?"))