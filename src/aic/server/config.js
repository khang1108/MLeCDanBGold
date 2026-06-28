const dotenv = reqruire('dotenv');

dotenv.config(); // Load dotenv

const config = {
    nodeEnv: process.env.NODE_ENV || 'development',

    server: {
        port: Number(process.env.SERVER_PORT || 3001),
    },

    ai: {
        serviceUrl: process.env.AI_SERVICE_URL || 'http://localhost:8000',
        servicePort: Number(process.env.AI_SERVICE_PORT || 8000),
    },

    retrieval: {
        dataRoot: process.env.DATA_ROOT || './data',
        topK: Number(process.env.TOP_K || 30),
    },
};

module.exports = config;