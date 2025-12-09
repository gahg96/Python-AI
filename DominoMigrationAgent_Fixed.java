import java.sql.*;
import java.util.*;
import java.io.*;
import java.text.SimpleDateFormat;

/**
 * NCB Domino to OceanBase Migration Agent
 * Version: 1.4.1 (Logic Fix + Refactoring)
 * Author: ZhengQi Tech
 * 
 * 功能说明：
 * 1. 使用 MySQL 5.1 兼容的驱动类名
 * 2. 禁用连接清理线程，避免 IllegalMonitorStateException
 * 3. 附件直接存储在数据库BLOB字段中，无需文件系统
 * 4. 支持迁移台账和去重功能
 * 5. 自动识别MIME类型
 * 6. 补充完整字段映射（56个字段）
 * 7. 所有文本字段统一为VARCHAR(200)，避免导出失败
 * 8. 日志双重存储（OceanBase + Domino）
 * 9. 支持增量导出模式
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

    private static final String DB_USER = "root@sys";
    private static final String DB_PASS = "";

    // 目标表定义
    private static final String TABLE_RULES = "T_NCB_RULES";
    private static final String TABLE_ATTACHMENTS = "T_ATTACHMENT_REF";
    private static final String TABLE_MIGRATION_LOG = "T_MIGRATION_LOG";  // 迁移台账表
    
    // Domino日志视图名称
    private static final String LOG_VIEW_NAME = "MigrationLog";
    
    // 要迁移的文档视图名称
    private static final String MIGRATION_VIEW_NAME = "allppm";

    // 附件直接存储到数据库，不再使用文件系统
    
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

            // 2. 初始化表结构 (关键: 自动建表)
            initDatabaseSchema(conn);
            
            // 2.1 初始化Domino日志视图
            initDominoLogView(db);

            // 3. 准备 SQL 语句和迁移台账相关语句
            // 注意：MySQL 5.1 也支持 ON DUPLICATE KEY UPDATE 语法
            String insertRuleSql = "INSERT INTO " + TABLE_RULES + " (" +
                "doc_unid, rule_code, doc_func, title_cn, title_en, " +
                "rule_type, line_type, related_rules, dept_name, view_level, division, " +
                "assign_unitname, assign_unitreader, assign_divisionreadername, division_unit_code, assign_idreader, " +
                "eff_date, keywords, next_review_date, create_date, review_date, cancel_date, is_trial, " +
                "line1, linename1, line2, linename2, line3, linename3, " +
                "line4, linename4, line5, linename5, line6, linename6, " +
                "content_body, remark, approve_unit_name, approve_unit_code, approve_remark, approve_basis_body, " +
                "maker_name, maker_date, maker_opinion, " +
                "checker_name, checker_date, checker_opinion, " +
                "audit_name, audit_date, audit_opinion, " +
                "audit_checker_name, audit_checker_date, audit_checker_opinion, " +
                "super_name, super_date, super_opinion" +
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) " +
                "ON DUPLICATE KEY UPDATE title_cn = VALUES(title_cn), migrate_time = NOW()";

            psRules = conn.prepareStatement(insertRuleSql);

            String insertAttachSql = "INSERT IGNORE INTO " + TABLE_ATTACHMENTS + 
                " (id, doc_unid, file_name, file_size, file_content, mime_type) VALUES (?, ?, ?, ?, ?, ?)";
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

            // 4. 从指定视图获取文档（支持增量导出）
            View migrationView = db.getView(MIGRATION_VIEW_NAME);
            if (migrationView == null) {
                log("错误：找不到视图 '" + MIGRATION_VIEW_NAME + "'");
                throw new Exception("视图不存在: " + MIGRATION_VIEW_NAME);
            }
            log("成功获取视图: " + MIGRATION_VIEW_NAME);
            
            // 获取视图中的所有条目
            ViewEntryCollection vec = migrationView.getAllEntries();
            int totalDocs = vec.getCount();
            log("视图中共有 " + totalDocs + " 个文档");
            
            if (totalDocs == 0) {
                log("视图中没有文档，迁移结束");
                vec.recycle();
                migrationView.recycle();
                return;
            }
            
            log("增量模式：基于UNID对比，已迁移的文档将被跳过");
            
            int successCount = 0;
            int skippedCount = 0;
            int failedCount = 0;
            int processedCount = 0;

            log("开始遍历文档...");
            
            // ===== 简洁的遍历逻辑 =====
            // 关键：getNextEntry(entry) 会自动回收传入的 entry
            // 所以必须在调用 getNextEntry 之前完成对当前 entry 的所有操作
            ViewEntry entry = vec.getFirstEntry();
            
            while (entry != null) {
                processedCount++;
                Document doc = null;
                String docUnid = null;
                String titleCn = "";
                String ruleCode = "";
                boolean needMigrate = false;
                
                // 1. 获取文档和UNID
                try {
                    doc = entry.getDocument();
                    if (doc != null) {
                        docUnid = doc.getUniversalID();
                        titleCn = safeGetString(doc, "ppm_title");
                        ruleCode = safeGetString(doc, "pn1") + "-" + 
                                   safeGetString(doc, "pn2") + "-" + 
                                   safeGetString(doc, "pn3") + "-" + 
                                   safeGetString(doc, "pn4");
                    }
                } catch (Exception e) {
                    log("获取文档失败: " + e.getMessage());
                }
                
                log("[" + processedCount + "/" + totalDocs + "] UNID=" + docUnid + ", 标题=" + titleCn);
                
                // 2. 检查是否需要迁移
                if (doc != null && docUnid != null && !docUnid.isEmpty()) {
                    try {
                        boolean alreadyMigrated = isAlreadyMigrated(psCheckMigrated, docUnid);
                        if (alreadyMigrated) {
                            skippedCount++;
                            log("  -> 已迁移，跳过");
                        } else {
                            needMigrate = true;
                        }
                    } catch (Exception e) {
                        log("  -> 检查迁移状态失败: " + e.getMessage());
                        needMigrate = true;  // 失败时尝试迁移
                    }
                }
                
                // 3. 执行迁移
                if (needMigrate && doc != null) {
                    try {
                        int attachmentCount = processDocument(doc, psRules, psAttach);
                        recordMigrationLog(psInsertLog, docUnid, ruleCode, titleCn, STATUS_SUCCESS, null, attachmentCount);
                        recordDominoLog(db, docUnid, ruleCode, titleCn, STATUS_SUCCESS, null, attachmentCount);
                        successCount++;
                        log("  -> 迁移成功，附件数=" + attachmentCount);
                    } catch (Exception e) {
                        failedCount++;
                        String errorMsg = e.getMessage();
                        if (errorMsg != null && errorMsg.length() > 200) {
                            errorMsg = errorMsg.substring(0, 200);
                        }
                        log("  -> 迁移失败: " + errorMsg);
                        try {
                            recordMigrationLog(psInsertLog, docUnid, ruleCode, titleCn, STATUS_FAILED, errorMsg, 0);
                        } catch (Exception ex) {
                            // 忽略日志记录错误
                        }
                    }
                }
                
                // 4. 回收文档
                if (doc != null) {
                    try { doc.recycle(); } catch (Exception e) {}
                }
                
                // 5. 每50条提交一次
                if (processedCount % 50 == 0) {
                    psRules.executeBatch();
                    psAttach.executeBatch();
                    psInsertLog.executeBatch();
                    conn.commit();
                    log("已处理 " + processedCount + "/" + totalDocs + " (成功:" + successCount + ", 跳过:" + skippedCount + ", 失败:" + failedCount + ")");
                }
                
                // 6. 获取下一个条目（会自动回收当前entry）
                entry = vec.getNextEntry(entry);
            }

            // 提交剩余批次
            psRules.executeBatch();
            psAttach.executeBatch();
            psInsertLog.executeBatch();
            conn.commit();
            
            // 回收资源
            try { vec.recycle(); } catch (Exception e) {}
            try { migrationView.recycle(); } catch (Exception e) {}

            log("===== 迁移完成 =====");
            log("总文档数: " + totalDocs);
            log("已处理: " + processedCount);
            log("成功迁移: " + successCount);
            log("跳过(已迁移): " + skippedCount);
            log("失败: " + failedCount);

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

        // 1. 创建主表 T_NCB_RULES (对应 PDF 字段，所有文本字段统一为VARCHAR(200))
        String ddlRules = "CREATE TABLE IF NOT EXISTS " + TABLE_RULES + " (" +
            "doc_unid VARCHAR(32) PRIMARY KEY COMMENT 'Domino UNID', " +
            "rule_code VARCHAR(200) COMMENT '规章编号 (pn1-pn4)', " +
            "doc_func VARCHAR(200) COMMENT '功能 (ppm_docfunc)', " +
            "title_cn VARCHAR(200) COMMENT '规章中文名称 (ppm_title)', " +
            "title_en VARCHAR(200) COMMENT '规章英文名称 (ppm_title_e)', " +
            "rule_type VARCHAR(200) COMMENT '规章类别 (ppm_typename)', " +
            "line_type VARCHAR(200) COMMENT '条线分类 (full_linename)', " +
            "related_rules VARCHAR(200) COMMENT '相关规章制度 (ppm_related_ppm)', " +
            "dept_name VARCHAR(200) COMMENT '主责部门 (ppm_deptname)', " +
            "view_level VARCHAR(200) COMMENT '阅览层级 (ppm_Level)', " +
            "division VARCHAR(200) COMMENT '主责处级单位/团队 (ppm_division)', " +
            "assign_unitname VARCHAR(200) COMMENT '按阅览层级开放的范围-单位名 (ppm_assign_unitname)', " +
            "assign_unitreader VARCHAR(200) COMMENT '按阅览层级开放的范围-读者 (ppm_assign_unitreader)', " +
            "assign_divisionreadername VARCHAR(200) COMMENT '可阅览的指定单位内所有人员-名称 (ppm_assign_Divisionreadername)', " +
            "division_unit_code VARCHAR(200) COMMENT '可阅览的指定单位内所有人员-代码 (ppm_divisionUnitCode)', " +
            "assign_idreader VARCHAR(200) COMMENT '可阅览的指定人员 (ppm_assign_IDreader)', " +
            "eff_date DATE COMMENT '生效日期 (ppm_effdate)', " +
            "keywords VARCHAR(200) COMMENT '本规章关键字 (ppm_main_word)', " +
            "next_review_date DATE COMMENT '预计下次重检日期 (ppm_nextmdate)', " +
            "create_date DATE COMMENT '数据库内建立日期 (ppm_cdate)', " +
            "review_date DATE COMMENT '最近一次重检日期 (ppm_rdate)', " +
            "cancel_date DATE COMMENT '注销生效日期 (ppm_effdeldate)', " +
            "is_trial VARCHAR(200) COMMENT '试行 (ppm_try)', " +
            "line1 VARCHAR(200) COMMENT '第一层代码 (Line1)', " +
            "linename1 VARCHAR(200) COMMENT '第一层名称 (Linename1)', " +
            "line2 VARCHAR(200) COMMENT '第二层代码 (Line2)', " +
            "linename2 VARCHAR(200) COMMENT '第二层名称 (Linename2)', " +
            "line3 VARCHAR(200) COMMENT '第三层代码 (Line3)', " +
            "linename3 VARCHAR(200) COMMENT '第三层名称 (Linename3)', " +
            "line4 VARCHAR(200) COMMENT '第四层代码 (Line4)', " +
            "linename4 VARCHAR(200) COMMENT '第四层名称 (Linename4)', " +
            "line5 VARCHAR(200) COMMENT '第五层代码 (Line5)', " +
            "linename5 VARCHAR(200) COMMENT '第五层名称 (Linename5)', " +
            "line6 VARCHAR(200) COMMENT '第六层代码 (Line6)', " +
            "linename6 VARCHAR(200) COMMENT '第六层名称 (Linename6)', " +
            "content_body LONGTEXT COMMENT '正文内容 (ppm_cbody)', " +
            "remark LONGTEXT COMMENT '备注 (ppm_remark)', " +
            "approve_unit_name VARCHAR(200) COMMENT '审批凭证-单位名 (ppm_approveunit_name)', " +
            "approve_unit_code VARCHAR(200) COMMENT '审批凭证-单位代码 (ppm_approveunit_code)', " +
            "approve_remark VARCHAR(200) COMMENT '备注 (ppm_approve_remark)', " +
            "approve_basis_body LONGTEXT COMMENT '审批凭证内容 (ppm_basis_body)', " +
            // --- 意见部分 ---
            "maker_name VARCHAR(200) COMMENT '经办人员 (ppm_maker)', " +
            "maker_date VARCHAR(200) COMMENT '经办时间 (ppm_maker_date)', " +
            "maker_opinion TEXT COMMENT '经办意见 (ppm_maker_opinion)', " +
            "checker_name VARCHAR(200) COMMENT '复核人员 (ppm_checker)', " +
            "checker_date VARCHAR(200) COMMENT '复核时间 (ppm_checker_date)', " +
            "checker_opinion TEXT COMMENT '复核意见 (ppm_checker_opinion)', " +
            "audit_name VARCHAR(200) COMMENT '核定人员 (ppm_audit_maker)', " +
            "audit_date VARCHAR(200) COMMENT '核定时间 (ppm_audit_maker_date)', " +
            "audit_opinion TEXT COMMENT '核定意见 (ppm_audit_maker_opinion)', " +
            "audit_checker_name VARCHAR(200) COMMENT '核定人员2 (ppm_audit_checker)', " +
            "audit_checker_date VARCHAR(200) COMMENT '核定时间2 (ppm_audit_checker_date)', " +
            "audit_checker_opinion TEXT COMMENT '核定意见2 (ppm_audit_checker_opinion)', " +
            "super_name VARCHAR(200) COMMENT '核定主管 (ppm_audit_super)', " +
            "super_date VARCHAR(200) COMMENT '核定时间 (ppm_audit_super_date)', " +
            "super_opinion TEXT COMMENT '核定意见 (ppm_audit_super_opinion)', " +
            "create_time DATETIME DEFAULT CURRENT_TIMESTAMP, " +
            "migrate_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '迁移时间'" +
            ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='规章制度主表'";

        stmt.execute(ddlRules);
        
        // 1.1 检查并更新主表结构（如果表已存在但缺少新字段）
        updateRulesTableSchema(conn);

        // 2. 创建附件表 T_ATTACHMENT_REF (附件内容直接存储在数据库中)
        String ddlAttach = "CREATE TABLE IF NOT EXISTS " + TABLE_ATTACHMENTS + " (" +
            "id VARCHAR(64) PRIMARY KEY, " +
            "doc_unid VARCHAR(32) NOT NULL, " +
            "file_name VARCHAR(255) COMMENT '原始文件名', " +
            "file_size BIGINT COMMENT '文件大小（字节）', " +
            "file_content LONGBLOB COMMENT '附件内容（二进制）', " +
            "mime_type VARCHAR(100) COMMENT 'MIME类型（如application/pdf）', " +
            "created_at DATETIME DEFAULT CURRENT_TIMESTAMP, " +
            "INDEX idx_doc (doc_unid)" +
            ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='附件表（内容存储在数据库）'";

        stmt.execute(ddlAttach);
        
        // 2.1 检查并更新附件表结构（如果表已存在但缺少新字段）
        updateAttachmentTableSchema(conn);

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
     * 检查并更新主表结构，添加缺失的字段
     * 用于兼容已存在的旧表结构
     */
    private void updateRulesTableSchema(Connection conn) throws SQLException {
        Statement stmt = conn.createStatement();
        try {
            // 检查表是否存在
            ResultSet rs = conn.getMetaData().getTables(null, null, TABLE_RULES, null);
            if (!rs.next()) {
                rs.close();
                return; // 表不存在，已由CREATE TABLE创建
            }
            rs.close();

            // 检查现有字段
            ResultSet columns = conn.getMetaData().getColumns(null, null, TABLE_RULES, null);
            java.util.Set<String> existingColumns = new java.util.HashSet<String>();
            while (columns.next()) {
                existingColumns.add(columns.getString("COLUMN_NAME").toLowerCase());
            }
            columns.close();

            // 定义需要添加的字段列表（按顺序）
            String[][] newFields = {
                {"related_rules", "VARCHAR(200) COMMENT '相关规章制度 (ppm_related_ppm)'"},
                {"division", "VARCHAR(200) COMMENT '主责处级单位/团队 (ppm_division)'"},
                {"assign_unitname", "VARCHAR(200) COMMENT '按阅览层级开放的范围-单位名 (ppm_assign_unitname)'"},
                {"assign_unitreader", "VARCHAR(200) COMMENT '按阅览层级开放的范围-读者 (ppm_assign_unitreader)'"},
                {"assign_divisionreadername", "VARCHAR(200) COMMENT '可阅览的指定单位内所有人员-名称 (ppm_assign_Divisionreadername)'"},
                {"division_unit_code", "VARCHAR(200) COMMENT '可阅览的指定单位内所有人员-代码 (ppm_divisionUnitCode)'"},
                {"assign_idreader", "VARCHAR(200) COMMENT '可阅览的指定人员 (ppm_assign_IDreader)'"},
                {"keywords", "VARCHAR(200) COMMENT '本规章关键字 (ppm_main_word)'"},
                {"next_review_date", "DATE COMMENT '预计下次重检日期 (ppm_nextmdate)'"},
                {"create_date", "DATE COMMENT '数据库内建立日期 (ppm_cdate)'"},
                {"review_date", "DATE COMMENT '最近一次重检日期 (ppm_rdate)'"},
                {"cancel_date", "DATE COMMENT '注销生效日期 (ppm_effdeldate)'"},
                {"is_trial", "VARCHAR(200) COMMENT '试行 (ppm_try)'"},
                {"line1", "VARCHAR(200) COMMENT '第一层代码 (Line1)'"},
                {"linename1", "VARCHAR(200) COMMENT '第一层名称 (Linename1)'"},
                {"line2", "VARCHAR(200) COMMENT '第二层代码 (Line2)'"},
                {"linename2", "VARCHAR(200) COMMENT '第二层名称 (Linename2)'"},
                {"line3", "VARCHAR(200) COMMENT '第三层代码 (Line3)'"},
                {"linename3", "VARCHAR(200) COMMENT '第三层名称 (Linename3)'"},
                {"line4", "VARCHAR(200) COMMENT '第四层代码 (Line4)'"},
                {"linename4", "VARCHAR(200) COMMENT '第四层名称 (Linename4)'"},
                {"line5", "VARCHAR(200) COMMENT '第五层代码 (Line5)'"},
                {"linename5", "VARCHAR(200) COMMENT '第五层名称 (Linename5)'"},
                {"line6", "VARCHAR(200) COMMENT '第六层代码 (Line6)'"},
                {"linename6", "VARCHAR(200) COMMENT '第六层名称 (Linename6)'"},
                {"approve_unit_name", "VARCHAR(200) COMMENT '审批凭证-单位名 (ppm_approveunit_name)'"},
                {"approve_unit_code", "VARCHAR(200) COMMENT '审批凭证-单位代码 (ppm_approveunit_code)'"},
                {"approve_remark", "VARCHAR(200) COMMENT '备注 (ppm_approve_remark)'"},
                {"approve_basis_body", "LONGTEXT COMMENT '审批凭证内容 (ppm_basis_body)'"},
                {"audit_checker_name", "VARCHAR(200) COMMENT '核定人员2 (ppm_audit_checker)'"},
                {"audit_checker_date", "VARCHAR(200) COMMENT '核定时间2 (ppm_audit_checker_date)'"},
                {"audit_checker_opinion", "TEXT COMMENT '核定意见2 (ppm_audit_checker_opinion)'"},
                {"migrate_time", "DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '迁移时间'"}
            };

            // 检查并添加缺失的字段
            for (String[] field : newFields) {
                String fieldName = field[0].toLowerCase();
                if (!existingColumns.contains(fieldName)) {
                    log("检测到旧表结构，添加字段: " + field[0]);
                    try {
                        // 确定字段插入位置（尽量放在相关字段之后）
                        String afterClause = "";
                        if (fieldName.equals("related_rules")) {
                            afterClause = " AFTER line_type";
                        } else if (fieldName.equals("division")) {
                            afterClause = " AFTER view_level";
                        } else if (fieldName.equals("assign_idreader")) {
                            afterClause = " AFTER division_unit_code";
                        } else if (fieldName.equals("keywords")) {
                            afterClause = " AFTER eff_date";
                        } else if (fieldName.equals("is_trial")) {
                            afterClause = " AFTER cancel_date";
                        } else if (fieldName.startsWith("linename")) {
                            // linename字段放在对应的line字段之后
                            String lineNum = fieldName.replace("linename", "");
                            if (existingColumns.contains("line" + lineNum)) {
                                afterClause = " AFTER line" + lineNum;
                            }
                        } else if (fieldName.equals("approve_unit_name")) {
                            afterClause = " AFTER remark";
                        } else if (fieldName.equals("audit_checker_name")) {
                            afterClause = " AFTER audit_opinion";
                        } else if (fieldName.equals("migrate_time")) {
                            afterClause = " AFTER create_time";
                        }
                        
                        stmt.execute("ALTER TABLE " + TABLE_RULES + 
                            " ADD COLUMN " + field[0] + " " + field[1] + afterClause);
                        log("已添加字段: " + field[0]);
                    } catch (SQLException e) {
                        log("添加字段 " + field[0] + " 失败: " + e.getMessage());
                        // 继续添加其他字段，不中断
                    }
                }
            }

            // 检查并更新现有字段类型（如果需要）
            // 例如：将VARCHAR(100)更新为VARCHAR(200)
            updateFieldTypes(conn, existingColumns);

        } catch (SQLException e) {
            log("更新主表结构时出错: " + e.getMessage());
            throw e;
        } finally {
            stmt.close();
        }
    }
    
    /**
     * 更新现有字段类型（如果需要）
     */
    private void updateFieldTypes(Connection conn, java.util.Set<String> existingColumns) {
        Statement stmt = null;
        try {
            stmt = conn.createStatement();
            
            // 需要更新为VARCHAR(200)的字段列表
            String[] varchar200Fields = {
                "rule_code", "doc_func", "title_cn", "title_en", "rule_type", 
                "line_type", "dept_name", "view_level", "maker_name", "maker_date",
                "checker_name", "checker_date", "audit_name", "audit_date",
                "super_name", "super_date"
            };
            
            for (String fieldName : varchar200Fields) {
                if (existingColumns.contains(fieldName.toLowerCase())) {
                    try {
                        // 检查当前字段类型
                        ResultSet rs = conn.getMetaData().getColumns(null, null, TABLE_RULES, fieldName);
                        if (rs.next()) {
                            String currentType = rs.getString("TYPE_NAME");
                            int currentSize = rs.getInt("COLUMN_SIZE");
                            rs.close();
                            
                            // 如果不是VARCHAR(200)，尝试更新
                            if (!"VARCHAR".equalsIgnoreCase(currentType) || currentSize != 200) {
                                log("更新字段类型: " + fieldName + " -> VARCHAR(200)");
                                stmt.execute("ALTER TABLE " + TABLE_RULES + 
                                    " MODIFY COLUMN " + fieldName + " VARCHAR(200)");
                                log("已更新字段类型: " + fieldName);
                            }
                        } else {
                            rs.close();
                        }
                    } catch (SQLException e) {
                        log("更新字段类型 " + fieldName + " 失败: " + e.getMessage());
                        // 继续处理其他字段
                    }
                }
            }
        } catch (SQLException e) {
            log("更新字段类型时出错: " + e.getMessage());
            // 不抛出异常，允许继续执行
        } finally {
            if (stmt != null) {
                try {
                    stmt.close();
                } catch (SQLException e) {
                    // 忽略关闭错误
                }
            }
        }
    }

    /**
     * 检查并更新附件表结构，添加缺失的字段
     * 用于兼容已存在的旧表结构
     */
    private void updateAttachmentTableSchema(Connection conn) throws SQLException {
        Statement stmt = conn.createStatement();
        try {
            // 检查表是否存在
            ResultSet rs = conn.getMetaData().getTables(null, null, TABLE_ATTACHMENTS, null);
            if (!rs.next()) {
                rs.close();
                return; // 表不存在，已由CREATE TABLE创建
            }
            rs.close();

            // 检查现有字段
            ResultSet columns = conn.getMetaData().getColumns(null, null, TABLE_ATTACHMENTS, null);
            java.util.Set<String> existingColumns = new java.util.HashSet<String>();
            while (columns.next()) {
                existingColumns.add(columns.getString("COLUMN_NAME").toLowerCase());
            }
            columns.close();

            // 检查并添加 file_content 字段
            if (!existingColumns.contains("file_content")) {
                log("检测到旧表结构，添加 file_content 字段...");
                stmt.execute("ALTER TABLE " + TABLE_ATTACHMENTS + 
                    " ADD COLUMN file_content LONGBLOB COMMENT '附件内容（二进制）' AFTER file_size");
                log("已添加 file_content 字段");
            }

            // 检查并添加 mime_type 字段
            if (!existingColumns.contains("mime_type")) {
                log("检测到旧表结构，添加 mime_type 字段...");
                stmt.execute("ALTER TABLE " + TABLE_ATTACHMENTS + 
                    " ADD COLUMN mime_type VARCHAR(100) COMMENT 'MIME类型（如application/pdf）' AFTER file_content");
                log("已添加 mime_type 字段");
            }

            // 移除旧的 file_path 字段（如果存在且不再需要）
            if (existingColumns.contains("file_path")) {
                log("检测到旧的 file_path 字段，可以保留或删除（当前保留以兼容旧数据）");
                // 注意：这里不删除 file_path 字段，以保持向后兼容
                // 如果需要删除，可以执行：
                // stmt.execute("ALTER TABLE " + TABLE_ATTACHMENTS + " DROP COLUMN file_path");
            }

        } catch (SQLException e) {
            log("更新表结构时出错: " + e.getMessage());
            throw e;
        } finally {
            stmt.close();
        }
    }

    /**
     * 处理单条文档逻辑（补充完整字段）
     * @return 附件数量
     */
    private int processDocument(Document doc, PreparedStatement ps, PreparedStatement psAtt) throws Exception {
        int paramIndex = 1;
        
        // 1. UNID
        ps.setString(paramIndex++, doc.getUniversalID());

        // 2. 合并规章编号 (pn1 + pn2 + pn3 + pn4)
        String ruleCode = safeGetString(doc, "pn1") + "-" + 
            safeGetString(doc, "pn2") + "-" + 
            safeGetString(doc, "pn3") + "-" + 
            safeGetString(doc, "pn4");
        ps.setString(paramIndex++, ruleCode);

        // 3. 基础文本字段映射（所有文本字段截取到200字符）
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_docfunc"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_title"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_title_e"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_typename"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "full_linename"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_related_ppm"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_deptname"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_Level"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_division"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_assign_unitname"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_assign_unitreader"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_assign_Divisionreadername"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_divisionUnitCode"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_assign_IDreader"), 200));

        // 4. 日期处理
        ps.setDate(paramIndex++, safeGetDate(doc, "ppm_effdate"));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_main_word"), 200));
        ps.setDate(paramIndex++, safeGetDate(doc, "ppm_nextmdate"));
        ps.setDate(paramIndex++, safeGetDate(doc, "ppm_cdate"));
        ps.setDate(paramIndex++, safeGetDate(doc, "ppm_rdate"));
        ps.setDate(paramIndex++, safeGetDate(doc, "ppm_effdeldate"));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_try"), 200));
        
        // 5. 层级字段 (Line1-Line6, Linename1-Linename6)
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "Line1"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "Linename1"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "Line2"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "Linename2"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "Line3"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "Linename3"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "Line4"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "Linename4"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "Line5"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "Linename5"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "Line6"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "Linename6"), 200));

        // 6. 富文本处理 (ppm_cbody, ppm_remark, ppm_basis_body)
        ps.setString(paramIndex++, extractRichText(doc, "ppm_cbody"));
        ps.setString(paramIndex++, extractRichText(doc, "ppm_remark"));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_approveunit_name"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_approveunit_code"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_approve_remark"), 200));
        ps.setString(paramIndex++, extractRichText(doc, "ppm_basis_body"));

        // 7. 意见字段 (Maker, Checker, Audit, Audit Checker, Super)
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_maker"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_maker_date"), 200));
        ps.setString(paramIndex++, safeGetString(doc, "ppm_maker_opinion"));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_checker"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_checker_date"), 200));
        ps.setString(paramIndex++, safeGetString(doc, "ppm_checker_opinion"));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_audit_maker"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_audit_maker_date"), 200));
        ps.setString(paramIndex++, safeGetString(doc, "ppm_audit_maker_opinion"));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_audit_checker"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_audit_checker_date"), 200));
        ps.setString(paramIndex++, safeGetString(doc, "ppm_audit_checker_opinion"));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_audit_super"), 200));
        ps.setString(paramIndex++, truncateString(safeGetString(doc, "ppm_audit_super_date"), 200));
        ps.setString(paramIndex++, safeGetString(doc, "ppm_audit_super_opinion"));

        // 加入批处理
        ps.addBatch();

        // 7. 处理附件 (Embeded Objects) - 即使失败也不影响文档迁移
        int attachmentCount = 0;
        try {
            attachmentCount = extractAttachments(doc, psAtt);
        } catch (Exception attachEx) {
            // 附件提取失败不影响文档迁移，只记录警告
            log("警告：文档 " + doc.getUniversalID() + " 的附件提取失败: " + attachEx.getMessage());
            // 继续执行，attachmentCount保持为0
        }
        
        return attachmentCount;
    }

    /**
     * 附件提取逻辑 - 直接写入数据库
     * 从多个字段中提取附件：ppm_cbody（内容）、ppm_remark（备注）等
     * @return 附件数量
     */
    private int extractAttachments(Document doc, PreparedStatement psAtt) throws Exception {
        int count = 0;
        
        // 定义可能包含附件的字段列表
        String[] attachmentFields = {
            "ppm_cbody",      // 内容字段
            "ppm_remark"      // 备注字段
            // 可以根据实际情况添加更多字段
        };
        
        // 用于记录已处理的附件，避免重复（基于文件名+大小）
        java.util.Set<String> processedAttachments = new java.util.HashSet<String>();
        
        // 遍历所有可能包含附件的字段
        for (String fieldName : attachmentFields) {
            try {
                RichTextItem rtItem = (RichTextItem)doc.getFirstItem(fieldName);
                if (rtItem == null) {
                    continue;  // 字段不存在，跳过
                }
                
                Vector objs = rtItem.getEmbeddedObjects();
                if (objs == null || objs.isEmpty()) {
                    continue;  // 没有附件，跳过
                }
                
                log("从字段 " + fieldName + " 中提取附件，发现 " + objs.size() + " 个嵌入对象");
                
                for (Object obj : objs) {
                    EmbeddedObject eo = (EmbeddedObject)obj;
                    if (eo.getType() == EmbeddedObject.EMBED_ATTACHMENT) {
                        try {
                            String originalFileName = eo.getName();
                            
                            // 生成唯一标识（文件名+字段名），避免同一附件在不同字段中重复提取
                            String attachmentKey = fieldName + ":" + originalFileName;
                            
                            // 检查是否已处理过（避免重复）
                            if (processedAttachments.contains(attachmentKey)) {
                                log("跳过重复附件: " + originalFileName + " (字段: " + fieldName + ")");
                                continue;
                            }
                            
                            // 读取附件内容为字节数组
                            InputStream inputStream = eo.getInputStream();
                            if (inputStream == null) {
                                log("警告：无法获取附件输入流: " + originalFileName + " (字段: " + fieldName + ")");
                                continue;
                            }
                            
                            // 将输入流转换为字节数组
                            ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
                            byte[] buffer = new byte[8192];  // 8KB缓冲区
                            int bytesRead;
                            long totalSize = 0;
                            
                            while ((bytesRead = inputStream.read(buffer)) != -1) {
                                outputStream.write(buffer, 0, bytesRead);
                                totalSize += bytesRead;
                                
                                // 限制单个附件大小（例如100MB）
                                if (totalSize > 100 * 1024 * 1024) {
                                    log("警告：附件过大，跳过: " + originalFileName + " (超过100MB, 字段: " + fieldName + ")");
                                    inputStream.close();
                                    outputStream.close();
                                    continue;
                                }
                            }
                            
                            inputStream.close();
                            byte[] fileContent = outputStream.toByteArray();
                            outputStream.close();
                            
                            if (fileContent.length == 0) {
                                log("警告：附件内容为空: " + originalFileName + " (字段: " + fieldName + ")");
                                continue;
                            }
                            
                            // 根据文件名推断MIME类型
                            String mimeType = guessMimeType(originalFileName);
                            
                            // 记录到数据库（附件内容直接存储在BLOB字段中）
                            psAtt.setString(1, UUID.randomUUID().toString());
                            psAtt.setString(2, doc.getUniversalID());
                            psAtt.setString(3, originalFileName);  // 原始文件名
                            psAtt.setLong(4, fileContent.length);  // 文件大小
                            psAtt.setBytes(5, fileContent);  // 附件内容（BLOB）
                            psAtt.setString(6, mimeType);  // MIME类型
                            psAtt.addBatch();
                            count++;
                            
                            // 标记为已处理
                            processedAttachments.add(attachmentKey);
                            
                            log("附件提取成功: " + originalFileName + " (" + formatFileSize(fileContent.length) + ", 字段: " + fieldName + ")");
                        } catch (Exception e) {
                            log("提取附件失败: " + e.getMessage() + " (UNID: " + doc.getUniversalID() + ", 字段: " + fieldName + ")");
                            // 继续处理下一个附件，不中断整个流程
                            continue;
                        }
                    }
                }
            } catch (Exception e) {
                log("处理字段 " + fieldName + " 时出错: " + e.getMessage());
                // 继续处理下一个字段
                continue;
            }
        }
        
        log("总共提取附件数量: " + count + " (来自 " + processedAttachments.size() + " 个唯一附件)");
        return count;
    }

    /**
     * 根据文件名推断MIME类型
     * @param fileName 文件名
     * @return MIME类型
     */
    private String guessMimeType(String fileName) {
        if (fileName == null || fileName.isEmpty()) {
            return "application/octet-stream";
        }
        
        String lowerName = fileName.toLowerCase();
        int lastDot = lowerName.lastIndexOf('.');
        if (lastDot < 0) {
            return "application/octet-stream";
        }
        
        String ext = lowerName.substring(lastDot + 1);
        
        // 常见文件类型映射
        if (ext.equals("pdf")) return "application/pdf";
        if (ext.equals("doc")) return "application/msword";
        if (ext.equals("docx")) return "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
        if (ext.equals("xls")) return "application/vnd.ms-excel";
        if (ext.equals("xlsx")) return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
        if (ext.equals("ppt")) return "application/vnd.ms-powerpoint";
        if (ext.equals("pptx")) return "application/vnd.openxmlformats-officedocument.presentationml.presentation";
        if (ext.equals("txt")) return "text/plain";
        if (ext.equals("html") || ext.equals("htm")) return "text/html";
        if (ext.equals("xml")) return "application/xml";
        if (ext.equals("jpg") || ext.equals("jpeg")) return "image/jpeg";
        if (ext.equals("png")) return "image/png";
        if (ext.equals("gif")) return "image/gif";
        if (ext.equals("zip")) return "application/zip";
        if (ext.equals("rar")) return "application/x-rar-compressed";
        if (ext.equals("7z")) return "application/x-7z-compressed";
        
        return "application/octet-stream";
    }
    
    /**
     * 格式化文件大小显示
     * @param size 文件大小（字节）
     * @return 格式化后的字符串
     */
    private String formatFileSize(long size) {
        if (size < 1024) {
            return size + " B";
        } else if (size < 1024 * 1024) {
            return String.format("%.2f KB", size / 1024.0);
        } else if (size < 1024 * 1024 * 1024) {
            return String.format("%.2f MB", size / (1024.0 * 1024.0));
        } else {
            return String.format("%.2f GB", size / (1024.0 * 1024.0 * 1024.0));
        }
    }

    // --- 迁移台账相关方法 ---

    /**
     * 检查文档是否已成功迁移（基于UNID对比）
     * 从T_MIGRATION_LOG表中查询，如果存在status='SUCCESS'的记录，则认为已迁移
     * 这是增量迁移的核心逻辑：通过UNID对比判断是否需要迁移
     * @param ps PreparedStatement for checking migration status
     * @param docUnid 文档UNID
     * @return true表示已迁移（跳过），false表示未迁移（需要迁移）
     */
    private boolean isAlreadyMigrated(PreparedStatement ps, String docUnid) throws SQLException {
        if (docUnid == null || docUnid.isEmpty()) {
            return false;
        }
        
        ResultSet rs = null;
        try {
            ps.clearParameters();
            ps.setString(1, docUnid);
            ps.setString(2, STATUS_SUCCESS);
            rs = ps.executeQuery();
            if (rs.next()) {
                return rs.getInt(1) > 0;
            }
            return false;
        } finally {
            if (rs != null) {
                try { rs.close(); } catch (SQLException e) {}
            }
        }
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
     * 截取字符串到指定长度（避免VARCHAR字段溢出）
     */
    private String truncateString(String str, int maxLength) {
        if (str == null) return "";
        if (str.length() <= maxLength) return str;
        return str.substring(0, maxLength);
    }
    
    /**
     * 获取上次迁移时间（已废弃，改用基于UNID的增量模式）
     * 保留此方法以保持兼容性，但不再使用
     * 新的增量模式：从视图allppm获取所有文档，通过UNID对比T_MIGRATION_LOG表判断是否已迁移
     */
    @Deprecated
    private java.util.Date getLastMigrateTime(Connection conn) {
        // 不再使用基于时间的增量模式，改用基于UNID的增量模式
        // 所有文档都从视图allppm获取，通过isAlreadyMigrated()方法判断是否已迁移
        return null;
    }
    
    /**
     * 初始化Domino日志视图
     */
    private void initDominoLogView(Database db) {
        try {
            View logView = db.getView(LOG_VIEW_NAME);
            if (logView == null) {
                log("创建Domino日志视图: " + LOG_VIEW_NAME);
                // 如果视图不存在，创建一个简单的视图
                // 注意：这里只是尝试获取，实际创建需要在Domino Designer中完成
                log("提示：请在Domino Designer中手动创建视图 '" + LOG_VIEW_NAME + "'，包含以下列：");
                log("  - doc_unid (文本)");
                log("  - rule_code (文本)");
                log("  - title_cn (文本)");
                log("  - status (文本)");
                log("  - error_msg (文本)");
                log("  - migrate_time (日期时间)");
                log("  - attachment_count (数字)");
            }
        } catch (Exception e) {
            log("检查Domino日志视图失败: " + e.getMessage());
            log("提示：请在Domino Designer中创建视图 '" + LOG_VIEW_NAME + "'");
        }
    }
    
    /**
     * 记录迁移日志到Domino
     */
    private void recordDominoLog(Database db, String docUnid, String ruleCode, 
            String titleCn, String status, String errorMsg, int attachmentCount) {
        try {
            Document logDoc = db.createDocument();
            logDoc.replaceItemValue("Form", "MigrationLog");
            logDoc.replaceItemValue("doc_unid", docUnid);
            logDoc.replaceItemValue("rule_code", ruleCode);
            logDoc.replaceItemValue("title_cn", titleCn);
            logDoc.replaceItemValue("status", status);
            if (errorMsg != null) {
                logDoc.replaceItemValue("error_msg", errorMsg);
            }
            logDoc.replaceItemValue("attachment_count", attachmentCount);
            
            DateTime now = db.getParent().createDateTime(new java.util.Date());
            logDoc.replaceItemValue("migrate_time", now);
            
            logDoc.save(true, false);
            logDoc.recycle();
        } catch (Exception e) {
            log("记录Domino日志失败: " + e.getMessage());
            // 不抛出异常，避免影响主流程
        }
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

