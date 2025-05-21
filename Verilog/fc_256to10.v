module fc_256to10 (
    input wire clk,
    input wire rst_n,
    input wire i_valid,
    input wire [255:0] i_data,
    input wire [255:0] i_weight,
    output reg [9:0] o_result,
    output reg o_valid
);

    integer i;

    // Stage 0
    reg [255:0] r_data_s0, r_weight_s0;
    reg         valid_s0;

    // Stage 1
    reg [255:0] r_xnor_s1;
    reg         valid_s1;

    // Add Tree
    reg valid_sum1;
    reg valid_sum2;
    reg valid_sum3;
    reg valid_sum4;
    reg valid_sum5;
    reg valid_sum6;
    reg valid_sum7;
    reg valid_result;

    reg [1:0] sum1 [0:127];
    reg [2:0] sum2 [0:63];
    reg [3:0] sum3 [0:31];
    reg [4:0] sum4 [0:15];
    reg [5:0] sum5 [0:7];
    reg [6:0] sum6 [0:3];
    reg [7:0] sum7 [0:1];
    reg [9:0] result;

    // Stage 2
    reg [9:0]   r_popcount_s2;
    reg         valid_s2;

    // ------------------------
    // Stage 0: Capture input
    // ------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_data_s0   <= 0;
            r_weight_s0 <= 0;
            valid_s0    <= 0;
        end else begin
            if (i_valid) begin
                r_data_s0    <= i_data;
                r_weight_s0  <= i_weight;
                valid_s0     <= 1;
            end else begin
                valid_s0 <= 0;
            end 
        end
    end

    // ------------------------
    // Stage 1: XNOR
    // ------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_xnor_s1 <= 0;
            valid_s1  <= 0;
        end else begin
            if (valid_s0) begin
                r_xnor_s1 <= ~(r_data_s0 ^ r_weight_s0);
                valid_s1  <= 1;
            end else begin
                valid_s1  <= 0;
            end
        end
    end

    // ------------------------
    // Stage 2: Popcount
    // ------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_sum1 <= 0;
            valid_sum2 <= 0;            
            valid_sum3 <= 0;
            valid_sum4 <= 0;
            valid_sum5 <= 0;
            valid_sum6 <= 0;
            valid_sum7 <= 0;
            valid_result <= 0;            
        end else begin
            valid_sum1  <= valid_s1;
            valid_sum2  <= valid_sum1 ;           
            valid_sum3  <= valid_sum2;
            valid_sum4  <= valid_sum3;
            valid_sum5  <= valid_sum4;
            valid_sum6  <= valid_sum5;
            valid_sum7  <= valid_sum6;
            valid_result <= valid_sum7;           
        end
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < 128; i = i + 1)
                sum1[i] <= 0;
            for (i = 0; i < 64; i = i + 1)
                sum2[i] <= 0;
            for (i = 0; i < 32; i = i + 1) 
                sum3[i] <= 0;
            for (i = 0; i < 16; i = i + 1)
                sum4[i] <= 0;
            for (i = 0; i < 8; i = i + 1)
                sum5[i] <= 0;
            for (i = 0; i < 4; i = i + 1)
                sum6[i] <= 0;
            for (i = 0; i < 2; i = i + 1)
                sum7[i] <= 0;

            result  <= 0;

        end else begin
            if (valid_sum1) begin
                for (i = 0; i < 128; i = i + 1) begin
                    sum1[i] <= r_xnor_s1[i] + r_xnor_s1[128+i];
                end
            end

            if (valid_sum2) begin
                for (i = 0; i < 64; i = i + 1) begin
                    sum2[i] <= sum1[i] + sum1[64+i];
                end
            end

            if (valid_sum3) begin
                for (i = 0; i < 32; i = i + 1) begin
                    sum3[i] <= sum2[i] + sum2[32+i];
                end
            end

            if (valid_sum4) begin
                for (i = 0; i < 16; i = i + 1) begin
                    sum4[i] <= sum3[i] + sum3[16+i];
                end
            end

            if (valid_sum5) begin
                for (i = 0; i < 8; i = i + 1) begin
                    sum5[i] <= sum4[i] + sum4[8+i];
                end
            end

            if (valid_sum6) begin
                for (i = 0; i < 4; i = i + 1) begin
                    sum6[i] <= sum5[i] + sum5[4+i];
                end
            end

            if (valid_sum7) begin
                for (i = 0; i < 2; i = i + 1) begin
                    sum7[i] <= sum6[i] + sum6[2+i];
                end
            end

            if (valid_result) begin
                result <= sum7[0] + sum7[1];
            end

        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_popcount_s2   <= 0;
            valid_s2     <= 0;
        end else begin
            if (valid_result) begin
                r_popcount_s2     <= result;
                valid_s2          <= 1;
            end else begin
                valid_s2 <= 0;
            end
        end
    end
    // ------------------------
    // Stage 3: Final compare
    // ------------------------

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_result <= 0;
            o_valid  <= 0;
        end else begin
            if (valid_s2) begin
                o_result <= r_popcount_s2;
                o_valid  <= 1;
            end else begin
                o_valid <= 0;
            end
        end
    end
endmodule