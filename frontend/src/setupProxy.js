/**
 * Dev-server middleware:
 * 1. Sets no-cache headers to stop aggressive proxy/CDN/browser caching.
 * 2. Injects a cache-busting timestamp on /static/js/bundle.js in HTML responses,
 *    guaranteeing that Cloudflare and browsers always fetch the latest bundle.
 */
module.exports = function (app) {
  app.use((req, res, next) => {
    res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');
    res.setHeader('Surrogate-Control', 'no-store');

    const originalEnd = res.end;
    res.end = function (chunk, encoding) {
      if (chunk && (res.getHeader('content-type') || '').includes('text/html')) {
        let content = typeof chunk === 'string' ? chunk : chunk.toString(encoding || 'utf8');
        if (content.includes('/static/js/bundle.js')) {
          content = content.replace(
            /\/static\/js\/bundle\.js(?:\?[^"]*)?/g,
            `/static/js/bundle.js?_v=${Date.now()}`
          );
          chunk = Buffer.from(content, 'utf8');
          res.setHeader('Content-Length', chunk.length);
        }
      }
      return originalEnd.call(this, chunk, encoding);
    };

    next();
  });
};
