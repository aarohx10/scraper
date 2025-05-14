# Research Scraper API

A comprehensive API for researching companies and individuals, capable of scraping and analyzing various document types.

## Features

- Company research with website analysis
- Person research with profile analysis
- Document processing (PDF, DOCX, XLSX, PPTX)
- Rate limiting and request queuing
- Asynchronous processing
- Result compression
- Health monitoring

## Prerequisites

- Python 3.11+
- Docker
- Docker Compose
- Hetzner Cloud account (for deployment)

## Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd research-scraper
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the development server:
```bash
python server.py
```

Or using Docker Compose:
```bash
docker-compose up --build
```

## API Endpoints

### POST /research
Create a new research request.

Request body:
```json
{
    "company_name": "Example Corp",
    "company_website": "https://example.com",
    "person_name": "John Doe",
    "person_profile_url": "https://linkedin.com/in/johndoe",
    "search_context": "Additional search context",
    "max_depth": 2,
    "include_pdfs": true,
    "include_docs": true,
    "include_spreadsheets": true,
    "include_presentations": true
}
```

### GET /status/{request_id}
Check the status of a research request.

### GET /health
Health check endpoint.

## Deployment to Hetzner

1. Set up your Hetzner Cloud server:
   - Create a new server
   - Install Docker
   - Configure firewall (allow port 8000)

2. Configure SSH access:
   - Add your SSH key to the server
   - Update `deploy.sh` with your server details

3. Deploy:
```bash
chmod +x deploy.sh
./deploy.sh
```

## Environment Variables

- `PORT`: Server port (default: 8000)
- `PYTHONUNBUFFERED`: Enable Python output buffering (default: 1)

## Monitoring

- Logs are stored in `/var/log/research-scraper`
- Health check endpoint available at `/health`
- Rate limiting: 100 requests per hour

## Security

- Non-root user in Docker container
- Rate limiting
- Input validation
- CORS configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License