"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.jwtConstants = void 0;
class jwtConstants {
    static secret = process.env.JWT_SECRET || 'secretKey_lead_to_cash_system';
}
exports.jwtConstants = jwtConstants;
//# sourceMappingURL=constants.js.map