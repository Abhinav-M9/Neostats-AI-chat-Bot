import requests
import os
from config.config import SERPAPI_API_KEY

def search_web(query, num_results=3):
    if not SERPAPI_API_KEY:
        print("SerpAPI key not found, skipping web search")
        return []
    
    try:
        url = "https://serpapi.com/search"
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": num_results
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        results = []
        for result in data.get("organic_results", []):
            results.append({
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                "link": result.get("link", "")
            })
        
        return results
    except Exception as e:
        print(f"Web search error: {e}")
        return []

def search_web_duckduckgo(query, num_results=3):
    try:
        import requests
        from bs4 import BeautifulSoup
        
        url = f"https://duckduckgo.com/html/?q={query}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        results = []
        search_results = soup.find_all('div', class_='result')
        
        for result in search_results[:num_results]:
            title_elem = result.find('a', class_='result__a')
            snippet_elem = result.find('div', class_='result__snippet')
            
            if title_elem and snippet_elem:
                results.append({
                    "title": title_elem.get_text().strip(),
                    "snippet": snippet_elem.get_text().strip(),
                    "link": title_elem.get('href', '')
                })
        
        return results
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        return []
