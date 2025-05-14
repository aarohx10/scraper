from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict
import uvicorn
import asyncio
import json
import gzip
from datetime import datetime
import logging
import os
from enhanced_scraper import main as scraper_main
from ratelimit import limits, RateLimitException
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rate limiting configuration
CALLS = 100  # Number of calls
RATE_LIMIT_PERIOD = 3600  # Time period in seconds (1 hour)

app = FastAPI(
    title="Research Scraper API",
    description="API for comprehensive research and analysis of companies and individuals",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ResearchRequest(BaseModel):
    # Company Information
    company_name: str
    company_website: Optional[HttpUrl] = None
    
    # Person Information
    person_name: Optional[str] = None
    person_profile_url: Optional[HttpUrl] = None
    
    # Search Context
    search_context: Optional[str] = None
    
    # Processing Options
    max_depth: Optional[int] = 2
    include_pdfs: Optional[bool] = True
    include_docs: Optional[bool] = True
    include_spreadsheets: Optional[bool] = True
    include_presentations: Optional[bool] = True

class ResearchResult(BaseModel):
    request_id: str
    status: str
    message: str
    company_info: Optional[Dict] = None
    person_info: Optional[Dict] = None
    related_documents: Optional[List[Dict]] = None
    sources_processed: Optional[List[str]] = None
    processing_time: Optional[float] = None
    compressed_data: Optional[bytes] = None

# Store for active research tasks
active_tasks = {}

def compress_results(results: Dict) -> bytes:
    """Compress the results using gzip."""
    json_str = json.dumps(results, ensure_ascii=False)
    return gzip.compress(json_str.encode('utf-8'))

@limits(calls=CALLS, period=RATE_LIMIT_PERIOD)
def check_rate_limit():
    """Check if the rate limit has been exceeded."""
    pass

async def process_research_request(request_id: str, request: ResearchRequest):
    """Process a research request in the background."""
    start_time = time.time()
    try:
        logger.info(f"Starting research request {request_id}")
        
        results = {
            'company_info': {},
            'person_info': {},
            'related_documents': [],
            'sources_processed': []
        }
        
        # Process company information
        if request.company_website:
            company_data = await scraper_main(
                query=f"{request.company_name} {request.company_website}",
                num_results=5
            )
            results['company_info'].update(company_data)
            results['sources_processed'].append(str(request.company_website))
        
        # Process person information
        if request.person_profile_url:
            person_data = await scraper_main(
                query=f"{request.person_name} {request.person_profile_url}",
                num_results=5
            )
            results['person_info'].update(person_data)
            results['sources_processed'].append(str(request.person_profile_url))
        
        # Process additional context
        if request.search_context:
            context_data = await scraper_main(
                query=request.search_context,
                num_results=5
            )
            results['related_documents'].extend(context_data)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Compress results
        compressed_data = compress_results(results)
        
        # Store results
        active_tasks[request_id] = {
            'status': 'completed',
            'results': results,
            'compressed_data': compressed_data,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Completed research request {request_id}")
        
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {str(e)}")
        active_tasks[request_id] = {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

@app.post("/research", response_model=ResearchResult)
async def create_research_request(request: ResearchRequest, background_tasks: BackgroundTasks):
    """Create a new research request."""
    try:
        check_rate_limit()
    except RateLimitException:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize task status
    active_tasks[request_id] = {
        'status': 'processing',
        'timestamp': datetime.now().isoformat()
    }
    
    # Start processing in background
    background_tasks.add_task(process_research_request, request_id, request)
    
    return ResearchResult(
        request_id=request_id,
        status="processing",
        message="Research request accepted and processing started"
    )

@app.get("/status/{request_id}", response_model=ResearchResult)
async def get_request_status(request_id: str):
    """Get the status of a research request."""
    if request_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Request not found")
    
    task = active_tasks[request_id]
    
    if task['status'] == 'completed':
        return ResearchResult(
            request_id=request_id,
            status="completed",
            message="Research request completed successfully",
            **task['results'],
            processing_time=task.get('processing_time'),
            compressed_data=task['compressed_data']
        )
    elif task['status'] == 'error':
        raise HTTPException(status_code=500, detail=task['error'])
    else:
        return ResearchResult(
            request_id=request_id,
            status="processing",
            message="Research request is still processing"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_tasks": len(active_tasks)
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True) 