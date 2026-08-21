const s3Config = () => ({
  bucket: process.env.REACT_APP_S3_BUCKET,
  region: process.env.REACT_APP_S3_REGION,
  accessKeyId: process.env.REACT_APP_AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.REACT_APP_AWS_SECRET_ACCESS_KEY,
  sessionToken: process.env.REACT_APP_AWS_SESSION_TOKEN,
});

const isSafeKeyPart = (part) => part && part !== '.' && part !== '..';

export const displayVideoId = (videoId) => {
  const parts = String(videoId || '').split('.').filter(Boolean);
  return parts[parts.length - 1] || 'Unknown video';
};

export const s3VideoObjectKey = (videoId) => {
  const vidStr = String(videoId || '');
  if (!vidStr || !isSafeKeyPart(vidStr)) return null;
  const folder = vidStr.split('_')[0];
  return `data/${folder}/${vidStr}.mp4`;
};

export const getS3VideoUrl = async (videoId, config = s3Config()) => {
  const bucket = config?.bucket?.trim();
  const region = config?.region?.trim();
  const accessKeyId = config?.accessKeyId?.trim();
  const secretAccessKey = config?.secretAccessKey?.trim();
  const key = s3VideoObjectKey(videoId);
  if (!bucket || !region || !accessKeyId || !secretAccessKey || !key) return null;

  const [{ GetObjectCommand, S3Client }, { getSignedUrl }] = await Promise.all([
    import('@aws-sdk/client-s3'),
    import('@aws-sdk/s3-request-presigner'),
  ]);
  const client = new S3Client({
    region,
    credentials: {
      accessKeyId,
      secretAccessKey,
      ...(config?.sessionToken ? { sessionToken: config.sessionToken } : {}),
    },
  });
  return getSignedUrl(client, new GetObjectCommand({ Bucket: bucket, Key: key }), {
    expiresIn: 15 * 60,
  });
};

export const timestampSeconds = (timestampMs) => {
  const timestamp = Number(timestampMs);
  return Number.isFinite(timestamp) && timestamp >= 0
    ? timestamp / 1000
    : null;
};
