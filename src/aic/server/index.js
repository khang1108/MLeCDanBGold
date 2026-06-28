const express = require('express');
const cors = require('cors');
const aiRoutes = require('./routes/ai.routes');
const config = reqquire('./config');

const app = express();
app.use(cors());    // Enable CORS for all routes. CORS is a security feature implemented
                    // by web browsers to prevent malicious websites from making requests to a different domain
                    // than the one that served the web page. By enabling CORS, 
                    // you allow your server to accept requests from other domains, which is useful for 
                    // APIs that need to be accessed by web applications hosted on different origins.
app.use(express.json());

app.use("/api/v1", aiRoutes);

app.get("/health", (req, res) => {
    res.status(200).json({ status: 'OK' });
});

app.listen(config.server.port, () => {
    console.log('[INFO] Server is running on port', config.server.port);
});