import {
  FileStorage,
  VideoResolution,
  VideoStreamingPlaylistType,
  type FileStorageType,
  type VideoStreamingPlaylistType_Type
} from '@peertube/peertube-models'
import { generateP2PMediaLoaderHash } from '@peertube/peertube-node-utils'
import { logger } from '@server/helpers/logger.js'
import { CONFIG } from '@server/initializers/config.js'
import { getHLSPrivateFileUrl, getObjectStoragePublicFileUrl } from '@server/lib/object-storage/index.js'
import { generateHLSMasterPlaylistFilename, generateHlsSha256SegmentsFilename } from '@server/lib/paths.js'
import { isVideoInPrivateDirectory } from '@server/lib/video-privacy.js'
import { VideoFileModel } from '@server/models/video/video-file.js'
import {
  MStreamingPlaylist,
  MStreamingPlaylistFiles,
  MStreamingPlaylistFilesVideo,
  MStreamingPlaylistVideo,
  MVideo,
  MVideoPrivacy
} from '@server/types/models/index.js'
import memoizee from 'memoizee'
import { join } from 'path'
import { Op, Transaction } from 'sequelize'
import { AllowNull, BelongsTo, Column, CreatedAt, DataType, Default, ForeignKey, HasMany, Is, Table, UpdatedAt } from 'sequelize-typescript'
import { isArrayOf } from '../../helpers/custom-validators/misc.js'
import { isVideoFileInfoHashValid } from '../../helpers/custom-validators/videos.js'
import {
  CONSTRAINTS_FIELDS,
  MEMOIZE_LENGTH,
  MEMOIZE_TTL,
  P2P_MEDIA_LOADER_PEER_VERSION,
  STATIC_PATHS,
  WEBSERVER
} from '../../initializers/constants.js'
import { VideoRedundancyModel } from '../redundancy/video-redundancy.js'
import { SequelizeModel, doesExist, throwIfNotValid } from '../shared/index.js'
import { VideoModel } from './video.js'

@Table({
  tableName: 'videoStreamingPlaylist',
  indexes: [
    {
      fields: [ 'videoId' ]
    },
    {
      fields: [ 'videoId', 'type' ],
      unique: true
    },
    {
      fields: [ 'p2pMediaLoaderInfohashes' ],
      using: 'gin'
    }
  ]
})
export class VideoStreamingPlaylistModel extends SequelizeModel<VideoStreamingPlaylistModel> {
  @CreatedAt
  createdAt: Date

  @UpdatedAt
  updatedAt: Date

  @AllowNull(false)
  @Column
  type: VideoStreamingPlaylistType_Type

  @AllowNull(false)
  @Column
  playlistFilename: string

  @AllowNull(true)
  @Column(DataType.STRING(CONSTRAINTS_FIELDS.VIDEOS.URL.max))
  playlistUrl: string

  @AllowNull(false)
  @Is('VideoStreamingPlaylistInfoHashes', value => throwIfNotValid(value, v => isArrayOf(v, isVideoFileInfoHashValid), 'info hashes'))
  @Column(DataType.ARRAY(DataType.STRING))
  p2pMediaLoaderInfohashes: string[]

  @AllowNull(false)
  @Column
  p2pMediaLoaderPeerVersion: number

  @AllowNull(true)
  @Column
  segmentsSha256Filename: string

  @AllowNull(true)
  @Column
  segmentsSha256Url: string

  @ForeignKey(() => VideoModel)
  @Column
  videoId: number

  @AllowNull(false)
  @Default(FileStorage.FILE_SYSTEM)
  @Column
  storage: FileStorageType

  @BelongsTo(() => VideoModel, {
    foreignKey: {
      allowNull: false
    },
    onDelete: 'CASCADE'
  })
  Video: Awaited<VideoModel>

  @HasMany(() => VideoFileModel, {
    foreignKey: {
      allowNull: true
    },
    onDelete: 'CASCADE'
  })
  VideoFiles: Awaited<VideoFileModel>[]

  @HasMany(() => VideoRedundancyModel, {
    foreignKey: {
      allowNull: false
    },
    onDelete: 'CASCADE',
    hooks: true
  })
  RedundancyVideos: Awaited<VideoRedundancyModel>[]

  static doesInfohashExistCached = memoizee(VideoStreamingPlaylistModel.doesInfohashExist.bind(VideoStreamingPlaylistModel), {
    promise: true,
    max: MEMOIZE_LENGTH.INFO_HASH_EXISTS,
    maxAge: MEMOIZE_TTL.INFO_HASH_EXISTS
  })

  static doesInfohashExist (infoHash: string) {
    // Don't add a LIMIT 1 here to prevent seq scan by PostgreSQL (not sure why id doesn't use the index when we add a LIMIT)
    const query = 'SELECT 1 FROM "videoStreamingPlaylist" WHERE "p2pMediaLoaderInfohashes" @> $infoHash'

    return doesExist({ sequelize: this.sequelize, query, bind: { infoHash: `{${infoHash}}` } }) // Transform infoHash in a PG array
  }

  static buildP2PMediaLoaderInfoHashes (playlistUrl: string, files: { height: number }[]) {
    const hashes: string[] = []

    const version = Math.abs(P2P_MEDIA_LOADER_PEER_VERSION)

    // https://github.com/Novage/p2p-media-loader/blob/master/p2p-media-loader-core/lib/p2p-media-manager.ts#L115
    for (let i = 0; i < files.length; i++) {
      hashes.push(generateP2PMediaLoaderHash(`v${version}-${playlistUrl}-main-${i}`))
    }

    // Audio only stream
    if (files.some(f => f.height === 0)) {
      hashes.push(generateP2PMediaLoaderHash(`v${version}-${playlistUrl}-secondary-0`))
    }

    logger.debug('Assigned P2P Media Loader info hashes', { playlistUrl, hashes })

    return hashes
  }

  static async listByIncorrectPeerVersion () {
    const rows = await VideoStreamingPlaylistModel.unscoped().findAll({
      raw: true,
      attributes: [ 'id' ],
      where: {
        p2pMediaLoaderPeerVersion: {
          [Op.ne]: P2P_MEDIA_LOADER_PEER_VERSION
        }
      }
    })

    return rows.map(r => r.id)
  }

  static loadWithVideoAndFiles (id: number) {
    const options = {
      include: [
        {
          model: VideoModel.unscoped(),
          required: true
        },
        {
          model: VideoFileModel.unscoped()
        }
      ]
    }

    return VideoStreamingPlaylistModel.findByPk<MStreamingPlaylistFilesVideo>(id, options)
  }

  static loadWithVideo (id: number, transaction?: Transaction) {
    const options = {
      include: [
        {
          model: VideoModel.unscoped(),
          required: true
        }
      ],
      transaction
    }

    return VideoStreamingPlaylistModel.findByPk(id, options)
  }

  static loadHLSByVideo (videoId: number, transaction?: Transaction): Promise<MStreamingPlaylist> {
    const options = {
      where: {
        type: VideoStreamingPlaylistType.HLS,
        videoId
      },
      transaction
    }

    return VideoStreamingPlaylistModel.findOne(options)
  }

  static loadHLSByVideoWithVideo (videoId: number, transaction?: Transaction): Promise<MStreamingPlaylistVideo> {
    const options = {
      where: {
        type: VideoStreamingPlaylistType.HLS,
        videoId
      },
      include: [
        {
          model: VideoModel.unscoped(),
          required: true
        }
      ],
      transaction
    }

    return VideoStreamingPlaylistModel.findOne(options)
  }

  static async loadOrGenerate (video: MVideo, transaction?: Transaction) {
    let playlist = await VideoStreamingPlaylistModel.loadHLSByVideo(video.id, transaction)
    let generated = false

    if (!playlist) {
      generated = true

      playlist = new VideoStreamingPlaylistModel({
        p2pMediaLoaderPeerVersion: P2P_MEDIA_LOADER_PEER_VERSION,
        type: VideoStreamingPlaylistType.HLS,
        storage: FileStorage.FILE_SYSTEM,
        p2pMediaLoaderInfohashes: [],
        playlistFilename: generateHLSMasterPlaylistFilename(video.isLive),
        segmentsSha256Filename: generateHlsSha256SegmentsFilename(video.isLive),
        videoId: video.id
      })

      await playlist.save({ transaction })
    }

    return { generated, playlist: Object.assign(playlist, { Video: video }) }
  }

  static doesOwnedVideoUUIDExist (videoUUID: string, storage: FileStorageType) {
    const query = `SELECT 1 FROM "videoStreamingPlaylist" ` +
      `INNER JOIN "video" ON "video"."id" = "videoStreamingPlaylist"."videoId" ` +
      `AND "video"."remote" IS FALSE AND "video"."uuid" = $videoUUID ` +
      `AND "storage" = $storage LIMIT 1`

    return doesExist({ sequelize: this.sequelize, query, bind: { videoUUID, storage } })
  }

  assignP2PMediaLoaderInfoHashes (video: MVideo, files: { height: number }[]) {
    const masterPlaylistUrl = this.getMasterPlaylistUrl(video)

    this.p2pMediaLoaderInfohashes = VideoStreamingPlaylistModel.buildP2PMediaLoaderInfoHashes(masterPlaylistUrl, files)
  }

  // ---------------------------------------------------------------------------

  getMasterPlaylistUrl (video: MVideo) {
    if (video.isOwned()) {
      if (this.storage === FileStorage.OBJECT_STORAGE) {
        return this.getMasterPlaylistObjectStorageUrl(video)
      }

      return WEBSERVER.URL + this.getMasterPlaylistStaticPath(video)
    }

    return this.playlistUrl
  }

  private getMasterPlaylistObjectStorageUrl (video: MVideo) {
    if (video.hasPrivateStaticPath() && CONFIG.OBJECT_STORAGE.PROXY.PROXIFY_PRIVATE_FILES === true) {
      return getHLSPrivateFileUrl(video, this.playlistFilename)
    }

    return getObjectStoragePublicFileUrl(this.playlistUrl, CONFIG.OBJECT_STORAGE.STREAMING_PLAYLISTS)
  }

  // ---------------------------------------------------------------------------

  getSha256SegmentsUrl (video: MVideo) {
    if (video.isOwned()) {
      if (!this.segmentsSha256Filename) return null

      if (this.storage === FileStorage.OBJECT_STORAGE) {
        return this.getSha256SegmentsObjectStorageUrl(video)
      }

      return WEBSERVER.URL + this.getSha256SegmentsStaticPath(video)
    }

    return this.segmentsSha256Url
  }

  private getSha256SegmentsObjectStorageUrl (video: MVideo) {
    if (video.hasPrivateStaticPath() && CONFIG.OBJECT_STORAGE.PROXY.PROXIFY_PRIVATE_FILES === true) {
      return getHLSPrivateFileUrl(video, this.segmentsSha256Filename)
    }

    return getObjectStoragePublicFileUrl(this.segmentsSha256Url, CONFIG.OBJECT_STORAGE.STREAMING_PLAYLISTS)
  }

  // ---------------------------------------------------------------------------

  hasAudioAndVideoSplitted (this: MStreamingPlaylistFiles) {
    // We need at least 2 files to have audio and video splitted
    if (this.VideoFiles.length === 1) return false

    let hasAudio = false
    let hasVideo = false

    for (const file of this.VideoFiles) {
      // File contains both streams: audio and video is not splitted
      if (file.hasAudio() && file.hasVideo()) return false

      if (file.resolution === VideoResolution.H_NOVIDEO) hasAudio = true
      else if (file.hasVideo()) hasVideo = true

      if (hasVideo && hasAudio) return true
    }

    return false
  }

  getStringType () {
    if (this.type === VideoStreamingPlaylistType.HLS) return 'hls'

    return 'unknown'
  }

  getTrackerUrls (baseUrlHttp: string, baseUrlWs: string) {
    return [ baseUrlWs + '/tracker/socket', baseUrlHttp + '/tracker/announce' ]
  }

  hasSameUniqueKeysThan (other: MStreamingPlaylist) {
    return this.type === other.type &&
      this.videoId === other.videoId
  }

  withVideo (video: MVideo) {
    return Object.assign(this, { Video: video })
  }

  // ---------------------------------------------------------------------------

  static getPlaylistFileStaticPath (video: MVideoPrivacy, filename: string) {
    if (isVideoInPrivateDirectory(video.privacy)) {
      return join(STATIC_PATHS.STREAMING_PLAYLISTS.PRIVATE_HLS, video.uuid, filename)
    }

    return join(STATIC_PATHS.STREAMING_PLAYLISTS.HLS, video.uuid, filename)
  }

  private getMasterPlaylistStaticPath (video: MVideoPrivacy) {
    return VideoStreamingPlaylistModel.getPlaylistFileStaticPath(video, this.playlistFilename)
  }

  private getSha256SegmentsStaticPath (video: MVideoPrivacy) {
    return VideoStreamingPlaylistModel.getPlaylistFileStaticPath(video, this.segmentsSha256Filename)
  }
}
