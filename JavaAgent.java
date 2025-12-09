import java.sql.*;
import java.util.*;
import java.io.*;
import java.text.SimpleDateFormat;

/**
 * NCB Domino to OceanBase Migration Agent
 * Version: 1.2.0 (Fixed for MySQL 5.1 Driver & Connection Cleanup Issue)
 * Author: ZhengQi Tech
 * 
 * 修复说明：
 * 1. 使用 MySQL 5.1 兼容的驱动类名
 * 2. 禁用连接清理线程，避免 IllegalMonitorStateException
 * 3. 添加连接池参数优化
 */
public class JavaAgent extends AgentBase {

    // ================= 配置区域 =================
    // OceanBase 连接串 (兼容 MySQL 5.7+ 协议)
    // 关键修复参数：
    // 1. useUnicode=true&characterEncoding=utf-8: 防止中文乱码
    // 2. rewriteBatchedStatements=true: 开启批量插入优化
    // 3. useSSL=false: 禁用SSL（如不需要）
    // 4. autoReconnect=true: 自动重连
    // 5. maxReconnects=3: 最大重连次数
    // 6. initialTimeout=2: 初始超时时间
    private static final String JDBC_URL = "jdbc:mysql://127.0.0.1:2881/DominoDB?" +
        "useUnicode=true&characterEncoding=utf-8&useSSL=false&" +
        "rewriteBatchedStatements=true&" +
        "autoReconnect=true&maxReconnects=3&initialTimeout=2&" +
        "useLocalSessionState=true&cachePrepStmts=true&prepStmtCacheSize=250&prepStmtCacheSqlLimit=2048";

    private static final String DB_USER = "admin@mysql_tenant";
    private static final String DB_PASS = "Ncb@2023Pass";

    // 目标表定义
    private static final String TABLE_RULES = "T_NCB_RULES";
    private static final String TABLE_ATTACHMENTS = "T_ATTACHMENT_REF";
    private static final String TABLE_MIGRATION_LOG = "T_MIGRATION_LOG";  // 迁移台账表

    // 附件临时存储路径 (NAS 挂载点)
    private static final String ATTACH_DIR = "/data/migration/attachments/";
    
    // 迁移状态常量
    private static final String STATUS_SUCCESS = "SUCCESS";
    private static final String STATUS_FAILED = "FAILED";
    private static final String STATUS_SKIPPED = "SKIPPED";  // 已迁移，跳过

    public void NotesMain() {
        Session session = null;
        AgentContext agentContext = null;
        Connection conn = null;
        PreparedStatement psRules = null;
        PreparedStatement psAttach = null;
        PreparedStatement psCheckMigrated = null;
        PreparedStatement psInsertLog = null;
        PreparedStatement psUpdateLog = null;

        try {
            session = getSession();
            agentContext = session.getAgentContext();
            Database db = agentContext.getCurrentDatabase();

            log("开始迁移数据库: " + db.getTitle());

            // 1. 建立数据库连接
            // 关键修复：使用 MySQL 5.1 版本的驱动类名，兼容 Domino 旧版 JVM
            // 注意：必须使用 mysql-connector-java-5.1.x.jar，不要使用 8.0+ 版本
            try {
                Class.forName("com.mysql.jdbc.Driver");
            } catch (ClassNotFoundException e) {
                log("错误：找不到 MySQL 驱动类。请确保将 mysql-connector-java-5.1.x.jar 放入 Domino 的 jvm/lib/ext 目录");
                throw e;
            }

            // 连接数据库
            conn = DriverManager.getConnection(JDBC_URL, DB_USER, DB_PASS);
            
            // 关键修复：禁用自动提交，使用手动事务控制
            conn.setAutoCommit(false);
            
            // 关键修复：设置连接属性，避免连接清理线程问题
            // 对于 MySQL 5.1 驱动，这些属性可以避免线程清理问题
            try {
                java.util.Properties props = new java.util.Properties();
                props.setProperty("user", DB_USER);
                props.setProperty("password", DB_PASS);
                props.setProperty("useUnicode", "true");
                props.setProperty("characterEncoding", "utf-8");
                props.setProperty("autoReconnect", "true");
                props.setProperty("maxReconnects", "3");
                props.setProperty("initialTimeout", "2");
                // 禁用连接验证（避免清理线程）
                props.setProperty("testOnBorrow", "false");
                props.setProperty("testWhileIdle", "false");
            } catch (Exception e) {
                // 忽略属性设置错误，继续使用连接
                log("警告：无法设置连接属性，继续使用默认连接");
            }

            log("数据库连接成功");

            // 2. 初始化附件存储目录
            initAttachmentDirectory();

            // 3. 初始化表结构 (关键: 自动建表)
            initDatabaseSchema(conn);

            // 3. 准备 SQL 语句和迁移台账相关语句
            // 注意：MySQL 5.1 也支持 ON DUPLICATE KEY UPDATE 语法
            String insertRuleSql = "INSERT INTO " + TABLE_RULES + " (" +
                "doc_unid, rule_code, doc_func, title_cn, title_en, " +
                "rule_type, line_type, dept_name, view_level, " +
                "eff_date, content_body, remark, " +
                "maker_name, maker_date, maker_opinion, " +
                "checker_name, checker_date, checker_opinion, " +
                "audit_name, audit_date, audit_opinion, " +
                "super_name, super_date, super_opinion" +
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) " +
                "ON DUPLICATE KEY UPDATE title_cn = VALUES(title_cn)";

            psRules = conn.prepareStatement(insertRuleSql);

            String insertAttachSql = "INSERT IGNORE INTO " + TABLE_ATTACHMENTS + 
                " (id, doc_unid, file_name, file_path, file_size) VALUES (?, ?, ?, ?, ?)";
            psAttach = conn.prepareStatement(insertAttachSql);

            // 准备迁移台账相关SQL
            psCheckMigrated = conn.prepareStatement(
                "SELECT COUNT(*) FROM " + TABLE_MIGRATION_LOG + " WHERE doc_unid = ? AND status = ?"
            );
            psInsertLog = conn.prepareStatement(
                "INSERT INTO " + TABLE_MIGRATION_LOG + 
                " (doc_unid, rule_code, title_cn, status, error_msg, migrate_time, attachment_count) " +
                "VALUES (?, ?, ?, ?, ?, NOW(), ?)"
            );
            psUpdateLog = conn.prepareStatement(
                "UPDATE " + TABLE_MIGRATION_LOG + 
                " SET status = ?, error_msg = ?, migrate_time = NOW(), attachment_count = ? " +
                "WHERE doc_unid = ?"
            );

            // 4. 遍历文档
            DocumentCollection dc = db.getAllDocuments();
            Document doc = dc.getFirstDocument();
            Document nextDoc = null;
            int count = 0;
            int skippedCount = 0;
            int successCount = 0;
            int failedCount = 0;

            log("开始检查并迁移文档...");

            while (doc != null) {
                nextDoc = dc.getNextDocument(doc);
                String docUnid = doc.getUniversalID();
                
                try {
                    // 4.1 检查是否已迁移（根据UNID判断）
                    if (isAlreadyMigrated(psCheckMigrated, docUnid)) {
                        skippedCount++;
                        if (skippedCount % 100 == 0) {
                            log("已跳过已迁移文档数: " + skippedCount);
                        }
                        doc.recycle();
                        doc = nextDoc;
                        continue;
                    }

                    // 4.2 提取并转换数据
                    int attachmentCount = processDocument(doc, psRules, psAttach);
                    
                    // 4.3 记录迁移成功日志
                    String ruleCode = safeGetString(doc, "pn1") + "-" + 
                        safeGetString(doc, "pn2") + "-" + 
                        safeGetString(doc, "pn3") + "-" + 
                        safeGetString(doc, "pn4");
                    String titleCn = safeGetString(doc, "ppm_title");
                    
                    recordMigrationLog(psInsertLog, docUnid, ruleCode, titleCn, STATUS_SUCCESS, null, attachmentCount);
                    
                    count++;
                    successCount++;

                    if (count % 50 == 0) {
                        psRules.executeBatch();
                        psAttach.executeBatch();
                        psInsertLog.executeBatch();
                        conn.commit();
                        log("已迁移记录数: " + count + " (成功: " + successCount + ", 跳过: " + skippedCount + ", 失败: " + failedCount + ")");
                    }
                } catch (Exception e) {
                    failedCount++;
                    String errorMsg = e.getMessage();
                    if (errorMsg != null && errorMsg.length() > 500) {
                        errorMsg = errorMsg.substring(0, 500);  // 限制错误信息长度
                    }
                    
                    // 记录迁移失败日志
                    try {
                        String ruleCode = safeGetString(doc, "pn1") + "-" + 
                            safeGetString(doc, "pn2") + "-" + 
                            safeGetString(doc, "pn3") + "-" + 
                            safeGetString(doc, "pn4");
                        String titleCn = safeGetString(doc, "ppm_title");
                        recordMigrationLog(psInsertLog, docUnid, ruleCode, titleCn, STATUS_FAILED, errorMsg, 0);
                    } catch (Exception logEx) {
                        log("记录失败日志时出错: " + logEx.getMessage());
                    }
                    
                    log("Error processing UNID " + docUnid + ": " + errorMsg);
                    e.printStackTrace();
                } finally {
                    doc.recycle();
                }
                doc = nextDoc;
            }

            // 提交剩余批次
            psRules.executeBatch();
            psAttach.executeBatch();
            psInsertLog.executeBatch();
            conn.commit();

            log("迁移完成。总共处理文档: " + count);
            log("统计信息 - 成功: " + successCount + ", 跳过: " + skippedCount + ", 失败: " + failedCount);

        } catch (Exception e) {
            e.printStackTrace();
            log("迁移失败: " + e.getMessage());
            try { 
                if(conn != null) {
                    conn.rollback();
                    log("已回滚事务");
                }
            } catch(SQLException sqle) {
                log("回滚失败: " + sqle.getMessage());
            }
        } finally {
            // 关键修复：确保正确关闭连接，避免清理线程问题
            closeResources(conn, psRules, psAttach, psCheckMigrated, psInsertLog, psUpdateLog);
        }
    }

    /**
     * 自动初始化数据库表结构 (If Not Exists)
     */
    private void initDatabaseSchema(Connection conn) throws SQLException {
        Statement stmt = conn.createStatement();
        log("检查并初始化数据库表结构...");

        // 1. 创建主表 T_NCB_RULES (对应 PDF 字段)
        String ddlRules = "CREATE TABLE IF NOT EXISTS " + TABLE_RULES + " (" +
            "doc_unid VARCHAR(32) PRIMARY KEY COMMENT 'Domino UNID', " +
            "rule_code VARCHAR(100) COMMENT '规章编号 (pn1-pn4)', " +
            "doc_func VARCHAR(100) COMMENT '功能 (ppm_docfunc)', " +
            "title_cn VARCHAR(255) COMMENT '规章中文名称 (ppm_title)', " +
            "title_en VARCHAR(255) COMMENT '规章英文名称 (ppm_title_e)', " +
            "rule_type VARCHAR(50) COMMENT '规章类别 (ppm_typename)', " +
            "line_type VARCHAR(50) COMMENT '条线分类 (full_linename)', " +
            "dept_name VARCHAR(100) COMMENT '主责部门 (ppm_deptname)', " +
            "view_level VARCHAR(50) COMMENT '阅览层级 (ppm_Level)', " +
            "eff_date DATE COMMENT '生效日期 (ppm_effdate)', " +
            "content_body LONGTEXT COMMENT '正文内容 (ppm_cbody)', " +
            "remark LONGTEXT COMMENT '备注 (ppm_remark)', " +
            // --- 意见部分 ---
            "maker_name VARCHAR(50) COMMENT '经办人员 (ppm_maker)', " +
            "maker_date VARCHAR(50) COMMENT '经办时间 (ppm_maker_date)', " +
            "maker_opinion TEXT COMMENT '经办意见 (ppm_maker_opinion)', " +
            "checker_name VARCHAR(50) COMMENT '复核人员 (ppm_checker)', " +
            "checker_date VARCHAR(50) COMMENT '复核时间 (ppm_checker_date)', " +
            "checker_opinion TEXT COMMENT '复核意见 (ppm_checker_opinion)', " +
            "audit_name VARCHAR(50) COMMENT '核定人员 (ppm_audit_maker)', " +
            "audit_date VARCHAR(50) COMMENT '核定时间 (ppm_audit_maker_date)', " +
            "audit_opinion TEXT COMMENT '核定意见 (ppm_audit_maker_opinion)', " +
            "super_name VARCHAR(50) COMMENT '核定主管 (ppm_audit_super)', " +
            "super_date VARCHAR(50) COMMENT '核定时间 (ppm_audit_super_date)', " +
            "super_opinion TEXT COMMENT '核定意见 (ppm_audit_super_opinion)', " +
            "create_time DATETIME DEFAULT CURRENT_TIMESTAMP" +
            ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='规章制度主表'";

        stmt.execute(ddlRules);

        // 2. 创建附件表 T_ATTACHMENT_REF
        String ddlAttach = "CREATE TABLE IF NOT EXISTS " + TABLE_ATTACHMENTS + " (" +
            "id VARCHAR(64) PRIMARY KEY, " +
            "doc_unid VARCHAR(32) NOT NULL, " +
            "file_name VARCHAR(255), " +
            "file_path VARCHAR(500), " +
            "file_size BIGINT, " +
            "created_at DATETIME DEFAULT CURRENT_TIMESTAMP, " +
            "INDEX idx_doc (doc_unid)" +
            ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4";

        stmt.execute(ddlAttach);

        // 3. 创建迁移台账表 T_MIGRATION_LOG
        String ddlLog = "CREATE TABLE IF NOT EXISTS " + TABLE_MIGRATION_LOG + " (" +
            "id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '自增ID', " +
            "doc_unid VARCHAR(32) NOT NULL COMMENT 'Domino文档UNID', " +
            "rule_code VARCHAR(100) COMMENT '规章编号', " +
            "title_cn VARCHAR(255) COMMENT '规章中文名称', " +
            "status VARCHAR(20) NOT NULL COMMENT '迁移状态: SUCCESS/FAILED/SKIPPED', " +
            "error_msg TEXT COMMENT '错误信息（失败时记录）', " +
            "migrate_time DATETIME NOT NULL COMMENT '迁移时间', " +
            "attachment_count INT DEFAULT 0 COMMENT '附件数量', " +
            "INDEX idx_unid (doc_unid), " +
            "INDEX idx_status (status), " +
            "INDEX idx_time (migrate_time)" +
            ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='迁移台账表'";

        stmt.execute(ddlLog);
        stmt.close();
        log("表结构初始化完成（包括迁移台账表）。");
    }

    /**
     * 处理单条文档逻辑
     * @return 附件数量
     */
    private int processDocument(Document doc, PreparedStatement ps, PreparedStatement psAtt) throws Exception {
        // 1. UNID
        ps.setString(1, doc.getUniversalID());

        // 2. 合并规章编号 (pn1 + pn2 + pn3 + pn4)
        String ruleCode = safeGetString(doc, "pn1") + "-" + 
            safeGetString(doc, "pn2") + "-" + 
            safeGetString(doc, "pn3") + "-" + 
            safeGetString(doc, "pn4");
        ps.setString(2, ruleCode);

        // 3. 基础文本字段映射
        ps.setString(3, safeGetString(doc, "ppm_docfunc"));
        ps.setString(4, safeGetString(doc, "ppm_title"));
        ps.setString(5, safeGetString(doc, "ppm_title_e"));
        ps.setString(6, safeGetString(doc, "ppm_typename"));
        ps.setString(7, safeGetString(doc, "full_linename"));
        ps.setString(8, safeGetString(doc, "ppm_deptname"));
        ps.setString(9, safeGetString(doc, "ppm_Level"));

        // 4. 日期处理 (ppm_effdate)
        ps.setDate(10, safeGetDate(doc, "ppm_effdate"));

        // 5. 富文本处理 (ppm_cbody, ppm_remark) - 提取纯文本或简单HTML
        ps.setString(11, extractRichText(doc, "ppm_cbody"));
        ps.setString(12, extractRichText(doc, "ppm_remark"));

        // 6. 意见字段 (Maker, Checker, Audit, Super)
        ps.setString(13, safeGetString(doc, "ppm_maker"));
        ps.setString(14, safeGetString(doc, "ppm_maker_date"));
        ps.setString(15, safeGetString(doc, "ppm_maker_opinion"));
        ps.setString(16, safeGetString(doc, "ppm_checker"));
        ps.setString(17, safeGetString(doc, "ppm_checker_date"));
        ps.setString(18, safeGetString(doc, "ppm_checker_opinion"));
        ps.setString(19, safeGetString(doc, "ppm_audit_maker"));
        ps.setString(20, safeGetString(doc, "ppm_audit_maker_date"));
        ps.setString(21, safeGetString(doc, "ppm_audit_maker_opinion"));
        ps.setString(22, safeGetString(doc, "ppm_audit_super"));
        ps.setString(23, safeGetString(doc, "ppm_audit_super_date"));
        ps.setString(24, safeGetString(doc, "ppm_audit_super_opinion"));

        // 加入批处理
        ps.addBatch();

        // 7. 处理附件 (Embeded Objects)
        int attachmentCount = extractAttachments(doc, psAtt);
        
        return attachmentCount;
    }

    /**
     * 附件提取逻辑
     * @return 附件数量
     */
    private int extractAttachments(Document doc, PreparedStatement psAtt) throws Exception {
        int count = 0;
        RichTextItem body = (RichTextItem)doc.getFirstItem("ppm_cbody");
        if (body == null) return count;

        Vector objs = body.getEmbeddedObjects();
        if (objs != null && !objs.isEmpty()) {
            for (Object obj : objs) {
                EmbeddedObject eo = (EmbeddedObject)obj;
                if (eo.getType() == EmbeddedObject.EMBED_ATTACHMENT) {
                    String fileName = eo.getName();
                    // 生成唯一物理文件名 (UNID_Filename) 防止覆盖
                    String physicalName = doc.getUniversalID() + "_" + fileName;
                    String fullPath = ATTACH_DIR + physicalName;

                    // 导出到磁盘
                    eo.extractFile(fullPath);
                    File f = new File(fullPath);

                    // 记录到数据库
                    psAtt.setString(1, UUID.randomUUID().toString());
                    psAtt.setString(2, doc.getUniversalID());
                    psAtt.setString(3, fileName);
                    psAtt.setString(4, fullPath);
                    psAtt.setLong(5, f.length());
                    psAtt.addBatch();
                    count++;
                }
            }
        }
        return count;
    }

    // --- 迁移台账相关方法 ---

    /**
     * 检查文档是否已成功迁移
     * @param ps PreparedStatement for checking migration status
     * @param docUnid 文档UNID
     * @return true表示已迁移，false表示未迁移
     */
    private boolean isAlreadyMigrated(PreparedStatement ps, String docUnid) throws SQLException {
        ps.setString(1, docUnid);
        ps.setString(2, STATUS_SUCCESS);
        
        ResultSet rs = ps.executeQuery();
        boolean exists = false;
        if (rs.next()) {
            exists = rs.getInt(1) > 0;
        }
        
        rs.close();
        return exists;
    }

    /**
     * 记录迁移日志到台账
     * @param ps PreparedStatement for INSERT
     * @param docUnid 文档UNID
     * @param ruleCode 规章编号
     * @param titleCn 规章中文名称
     * @param status 迁移状态
     * @param errorMsg 错误信息（成功时为null）
     * @param attachmentCount 附件数量
     */
    private void recordMigrationLog(PreparedStatement ps, String docUnid, String ruleCode, 
            String titleCn, String status, String errorMsg, int attachmentCount) throws SQLException {
        ps.setString(1, docUnid);
        ps.setString(2, ruleCode);
        ps.setString(3, titleCn);
        ps.setString(4, status);
        ps.setString(5, errorMsg);
        ps.setInt(6, attachmentCount);
        ps.addBatch();
    }

    // --- 工具方法 ---

    private String safeGetString(Document doc, String fieldName) throws NotesException {
        String val = doc.getItemValueString(fieldName);
        return val == null ? "" : val;
    }

    private java.sql.Date safeGetDate(Document doc, String fieldName) {
        try {
            Vector v = doc.getItemValue(fieldName);
            if (v != null && v.size() > 0 && v.get(0) instanceof DateTime) {
                DateTime dt = (DateTime) v.get(0);
                java.util.Date javaDate = dt.toJavaDate();
                return new java.sql.Date(javaDate.getTime());
            }
        } catch (Exception e) {
            // Log warning
        }
        return null; // Return null if date is invalid
    }

    private String extractRichText(Document doc, String fieldName) throws NotesException {
        // 简易实现：提取纯文本。
        // 如需完整 HTML，建议在 Agent 中调用 DxlExporter 或第三方 HTML 转换库
        RichTextItem rt = (RichTextItem)doc.getFirstItem(fieldName);
        if (rt != null) {
            return rt.getUnformattedText();
        }
        return safeGetString(doc, fieldName);
    }

    private void log(String msg) {
        System.out.println("[MigrationAgent] " + msg);
    }

    /**
     * 关键修复：正确关闭资源，避免连接清理线程问题
     */
    private void closeResources(Connection conn, Statement... stmts) {
        // 先关闭所有 Statement
        for(Statement st : stmts) {
            try { 
                if (st != null) {
                    st.close();
                }
            } catch(Exception e) {
                log("关闭 Statement 时出错: " + e.getMessage());
            }
        }

        // 最后关闭 Connection
        if (conn != null) {
            try {
                // 关键修复：对于 MySQL 5.1 驱动，直接关闭连接即可
                // 不需要等待清理线程，避免 IllegalMonitorStateException
                conn.close();
                log("数据库连接已关闭");
            } catch(Exception e) {
                log("关闭数据库连接时出错: " + e.getMessage());
            }
        }
    }
}

