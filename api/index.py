from http.server import BaseHTTPRequestHandler
import json
import logging
import time
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def search_google(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Search Google and return results."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
        }
        response = requests.get(
            f'https://www.google.com/search?q={query}',
            headers=headers,
            timeout=5
        )
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        for g in soup.find_all('div', class_='g')[:num_results]:
            title = g.find('h3')
            link = g.find('a')
            snippet = g.find('div', class_='VwiC3b')
            
            if title and link and snippet:
                results.append({
                    'title': title.text,
                    'url': link['href'],
                    'snippet': snippet.text
                })
        
        return results
    except Exception as e:
        logger.error(f"Error in search_google: {str(e)}")
        return []

class handler(BaseHTTPRequestHandler):
    async def do_POST(self):
        start_time = time.time()
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Process the request
            results = await search_google(
                query=request_data.get('query', ''),
                num_results=request_data.get('num_results', 5)
            )
            
            # Add timing information
            response_data = {
                'status': 'success',
                'query': request_data.get('query', ''),
                'results': results,
                'total_time': time.time() - start_time
            }
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode())
            
        except json.JSONDecodeError:
            self.send_error_response(400, "Invalid JSON in request")
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            self.send_error_response(500, str(e))
    
    def send_error_response(self, status_code: int, error_message: str):
        """Send an error response with timing information."""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({
            'status': 'error',
            'error': error_message,
            'total_time': time.time() - start_time
        }).encode())

    async def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'status': 'healthy',
                'timestamp': time.time()
            }).encode())
        else:
            self.send_response(404)
            self.end_headers() 