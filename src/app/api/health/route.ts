// Health check endpoint for monitoring and deployment verification
export async function GET() {
  try {
    // Check if the application is running properly
    const healthCheck = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      environment: process.env.NODE_ENV || 'development',
      version: process.env.npm_package_version || '1.0.0',
      services: {
        api: 'operational',
        database: 'not_configured', // No database in this app
        ai_service: process.env.GOOGLE_GENERATIVE_AI_API_KEY ? 'configured' : 'not_configured'
      }
    };

    return new Response(JSON.stringify(healthCheck), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache, no-store, must-revalidate'
      }
    });

  } catch (error) {
    const errorResponse = {
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: error instanceof Error ? error.message : 'Unknown error',
      services: {
        api: 'error',
        database: 'not_configured',
        ai_service: 'unknown'
      }
    };

    return new Response(JSON.stringify(errorResponse), {
      status: 500,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache, no-store, must-revalidate'
      }
    });
  }
}
